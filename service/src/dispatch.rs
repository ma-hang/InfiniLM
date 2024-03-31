﻿use crate::{
    batcher::{Batcher, Task},
    session::{Command, SessionContext},
};
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{Arc, Mutex},
};
use tokio::{
    sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
    task::{spawn_blocking, JoinSet},
};
use transformer::{SampleArgs, Transformer};

pub fn run<T>(
    transformer: T,
    sample: Arc<Mutex<SampleArgs>>,
    commands: UnboundedReceiver<Command>,
) -> JoinSet<()>
where
    T: Transformer + Send + Sync + 'static,
    T::Cache: Send + 'static,
{
    let mut dispatcher = Dispatcher::new(transformer, Batcher::new());
    let messages = dispatcher.manage();
    dispatcher.forward(messages.clone(), commands);
    dispatcher.decode(sample, messages);

    dispatcher.set
}

struct Dispatcher<T: Transformer> {
    transformer: Arc<T>,
    batcher: Arc<Batcher<T::Cache>>,
    set: JoinSet<()>,
}

enum Message<Cache> {
    Cmd(Command),
    Ctx(SessionContext<Cache>),
}

impl<T> Dispatcher<T>
where
    T: Transformer + Send + Sync + 'static,
    T::Cache: Send + 'static,
{
    #[inline]
    pub fn new(transformer: T, batcher: Batcher<T::Cache>) -> Self {
        Dispatcher {
            transformer: Arc::new(transformer),
            batcher: Arc::new(batcher),
            set: JoinSet::new(),
        }
    }

    pub fn forward(
        &mut self,
        messages: UnboundedSender<Message<T::Cache>>,
        mut commands: UnboundedReceiver<Command>,
    ) {
        self.set.spawn(async move {
            while let Some(msg) = commands.recv().await {
                messages.send(Message::Cmd(msg)).unwrap();
            }
        });
    }

    pub fn manage(&mut self) -> UnboundedSender<Message<T::Cache>> {
        let (sender, mut receiver) = unbounded_channel();
        let transformer = self.transformer.clone();
        let batcher = self.batcher.clone();
        self.set.spawn(async move {
            let mut sessions = HashMap::new();
            let mut removing = HashSet::new();
            while let Some(msg) = receiver.recv().await {
                match msg {
                    Message::Cmd(Command::Infer(id, infer)) => {
                        let ctx = match sessions.entry(id) {
                            Entry::Occupied(ctx) => ctx.remove(),
                            Entry::Vacant(_) => SessionContext::new(transformer.new_cache(), id),
                        };
                        batcher.enq(Task { ctx, infer });
                    }
                    Message::Cmd(Command::Drop(id)) => {
                        if sessions.remove(&id).is_none() {
                            removing.insert(id);
                        }
                    }
                    Message::Ctx(ctx) => {
                        if !removing.remove(&ctx.id) {
                            sessions.insert(ctx.id, ctx);
                        }
                    }
                }
            }
        });
        sender
    }

    pub fn decode(
        &mut self,
        sample: Arc<Mutex<SampleArgs>>,
        sender: UnboundedSender<Message<T::Cache>>,
    ) {
        let max_seq_len = self.transformer.model().max_position_embeddings();
        let eos = self.transformer.model().eos_token_id();
        let transformer = self.transformer.clone();
        let batcher = self.batcher.clone();
        self.set.spawn_blocking(move || loop {
            let mut tasks = batcher.deq();

            let requests = tasks
                .iter_mut()
                .map(|task| task.ctx.request(&task.infer.prompt, max_seq_len))
                .collect::<Vec<_>>();

            let (requests, logits) = transformer.decode(requests);
            let transformer = transformer.clone();
            let sender = sender.clone();
            let batcher = batcher.clone();
            let sample = sample.clone();
            spawn_blocking(move || {
                let tokens = transformer
                    .sample(&sample.lock().unwrap(), requests, logits)
                    .into_iter()
                    .collect::<HashMap<_, _>>();

                for mut task in tasks {
                    match tokens.get(&task.ctx.id) {
                        Some(&token) => {
                            if token != eos {
                                task.infer.responsing.send(token).unwrap();
                                task.infer.prompt = vec![token];
                                batcher.enq(task);
                            } else {
                                sender.send(Message::Ctx(task.ctx)).unwrap();
                            }
                        }
                        None => todo!(),
                    };
                }
            });
        });
    }
}
