sms_raw <- read.csv("C:/eduCBA/naive_bayes/SMSSpamCollection.csv", stringsAsFactors = FALSE)
str(sms_raw)
colnames(sms_raw) <- c("type","text")
sms_raw$type <- ifelse(sms_raw$type == "ham\t","ham",sms_raw$type)
summary(sms_raw)
sms_raw$type <- as.factor(sms_raw$type)
table(sms_raw$type)
View(sms_raw)
library(tm)
sms_corpus <- Corpus(VectorSource(sms_raw$text))
sms_corpus
print(sms_corpus)
inspect(sms_corpus[1:3]) 
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
corpus_clean <- tm_map(corpus_clean, PlainTextDocument)
sms_dtm <- TermDocumentMatrix(corpus_clean,control = list(removePunctuation = TRUE, 
                                                          +                                                             stopwords = TRUE))

sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:nrow(sms_raw), ]
nrow(sms_raw_train)
nrow(sms_raw_test)
sms_dtm_train <- sms_dtm[1:4169, ]

sms_dtm_test  <- sms_dtm[4170:nrow(sms_raw), ]
sms_corpus_train <- corpus_clean[1:4169]


sms_corpus_test  <- corpus_clean[4170: nrow(sms_raw)]
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))
library(wordcloud)
wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)
spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))
findFreqTerms(sms_dtm_train, 6)
sms_dict <- Dictionary(findFreqTerms(sms_dtm_train,6))
#install.packages("tractor.base")
#library(tractor.base)
sms_dict <- findFreqTerms(sms_dtm_train, 6)
sms_dict
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))

convert_counts <- function(x) {
  x	<- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes")) 
  return(x)
}
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)
library(e1071) 
sms_classifier <- naiveBayes(data= sms_train, sms_raw_train$type)
sms_classifier
sms_test_pred <- predict(sms_classifier, sms_test)
library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type, prop.chisq = FALSE, prop.t = FALSE,
             dnn = c('predicted', 'actual'))
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_raw_test$type, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))