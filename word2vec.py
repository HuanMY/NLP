import gensim
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
class SentenceIterator:
    def __init__(self,data):
        self.data = data
    def __iter__(self):
        for ind,row in enumerate(self.data):
            yield row.strip().split(" ")
def word2vec(train):
    vector_size = 300
    savedir = '/home/kesci/work/'
    # 将整个过程分成三步
    # 1, 构建模型(不训练)
    model_word2vec = gensim.models.Word2Vec(min_count=1, 
                                            window=10, 
                                            size=vector_size,
                                            workers=4,
                                            batch_words=1000)
    # 2, 遍历一遍语料库
    since = time.time()
    sentences = SentenceIterator(list(train['query'])+list(train['title'])) # 你的sentence迭代器
    model_word2vec.build_vocab(sentences, progress_per=20000000)
    time_elapsed = time.time() - since
    print('Time to build vocab: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 3, 训练
    since = time.time()
    model_word2vec.train(sentences, total_examples=model_word2vec.corpus_count, 
                            epochs=30, compute_loss=True, report_delay=60*10, # 每隔10分钟输出一下日志
                            callbacks=[EpochSaver(savedir)])
    time_elapsed = time.time() - since
    print('Time to train: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_word2vec
