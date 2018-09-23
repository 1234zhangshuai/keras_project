
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# word_index是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index()
# 键值颠倒， 将整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 将评论解码。注意，索引减去了3，因为0,1,2是为“padding”(填充)，“star of sequence”（序列开始），“unknown”(未知词)分别保留的索引
decoded_review = ''.join([reverse_word_index.get(i - 3, '?') for i in train_data([0])])