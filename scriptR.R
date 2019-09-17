#Ferrario Marco 
#Giorgio Ottolina


#Per eseguire lo script sono utilizzati il dataset "master", ottenuto dallo script preparation visto a lezione,per l'email
#engagement, il dataset numero 7 per la parte di RFM.



library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(magrittr)
library(ggplot2)
library(forcats)
library(MLmetrics)
library(bnlearn)
library(ROSE)
library(randomForest)
library(ROCR)


dir = "C:/Users/marco/Downloads/DS_Lab_digital_marketing"
setwd(dir)

set.seed(12345)

df_email <- read.csv2("master.csv") #importo il dataset risultante dallo script preparation 

df_email <- df_email[,-c(1,8,9,10,11,18,21,23)]  # elimino le colonne con troppi factor ( > 50) o di tipo non supportato per i modelli

df_email <- df_email[-1] # elimino la prima colonna che permette di identificare univocamente il cliente

str(df_email) # controllo la tipologia delle feature


#trasformo le colonne che non lo sono in factor, poichè alcuni dei modelli utilizzati supportano solamente questo tipo

df_email$TARGET <- as.factor(df_email$TARGET)
df_email$NUM_SEND_PREV <- as.factor(df_email$NUM_SEND_PREV)
df_email$NUM_OPEN_PREV <- as.factor(df_email$NUM_OPEN_PREV)
df_email$NUM_CLICK_PREV <- as.factor(df_email$NUM_CLICK_PREV)
df_email$NUM_FAIL_PREV <- as.factor(df_email$NUM_FAIL_PREV)
df_email$ID_NEG <- as.factor(df_email$ID_NEG)
df_email$TYP_CLI_FID <- as.factor(df_email$TYP_CLI_FID)
df_email$STATUS_FID <- as.factor(df_email$STATUS_FID)
df_email$NUM_FIDs <- as.factor(df_email$NUM_FIDs)
df_email$TYP_CLI_ACCOUNT <- as.factor(df_email$TYP_CLI_ACCOUNT)
df_email$FLAG_PRIVACY_1 <- as.factor(df_email$FLAG_PRIVACY_1)
df_email$FLAG_PRIVACY_2 <- as.factor(df_email$FLAG_PRIVACY_2)
df_email$FLAG_DIRECT_MKT <- as.factor(df_email$FLAG_DIRECT_MKT)
df_email$W_PHONE <- as.factor(df_email$W_PHONE)


#cambio gli na
df_email$W_PHONE <- fct_explicit_na(df_email$W_PHONE, na_level = "0")  
df_email$EMAIL_PROVIDER_CLEAN <- fct_explicit_na(df_email$EMAIL_PROVIDER_CLEAN, na_level = "(Missing)")



str(df_email)

prop.table(table(df_email$TARGET)) #controllando le percentuali della frequenza della variabile target si nota lo sbilanciamento



training_indices <-  df_email$TARGET %>% createDataPartition(p=0.66, list = FALSE, times = 1) # creo la partition del training set


training_set <- df_email[training_indices, ] # subset del database email che sarà usata come training set

training_set <- ovun.sample(TARGET~.,training_set,method="both",p=0.5)$data # oversample del training set per lo sbilanciamento

test_set <- df_email[-training_indices,] # subset del database email che sarà usata come test set


decision_tree <- rpart(TARGET ~ ., data= training_set) #  alleno il modello del decision tree


pred <- rpart.predict(decision_tree, test_set[,-1],type = "class") # previsione sul test set

# risultati del modello
confusionMatrix(pred, test_set[,1],positive='1') 
recall(pred,test_set[,1],relevant = '1')
precision(pred,test_set[,1],relevant = '1')
F1_DT <- F1_Score(pred,test_set[,1],positive = '1')



# curva ROC
pred_with_prob <- rpart.predict(decision_tree, test_set[, -1], type = "prob")[,2]

ROC1 <- performance(prediction(pred_with_prob, test_set$TARGET), 'tpr', 'fpr')
ROC_DT <- data.frame(x=ROC1@x.values[[1]], y=ROC1@y.values[[1]])
AUC_DT <- round(performance(prediction(pred_with_prob, 
                                       test_set$TARGET),'auc')@y.values[[1]],3)

#random forest

random_forest <- randomForest(TARGET ~ ., data= training_set, ntree = 100)

pred_rf <- rpart.predict(random_forest, test_set[,-1], type = "class")


#risultati del modello

confusionMatrix(pred_rf, test_set[,1],positive='1')
recall(pred_rf, test_set[,1],relevant = '1')
precision(pred_rf ,test_set[,1],relevant = '1')
F1_RF <- F1_Score(test_set[,1],pred_rf,positive = '1')


pred_with_prob_rf <- rpart.predict(random_forest, test_set[, -1], type = "prob")[,2]

ROC2 <- performance(prediction(pred_with_prob_rf, test_set$TARGET),  
                    measure='tpr', x.measure='fpr')
ROC_RF <- data.frame(x=ROC2@x.values[[1]], y=ROC2@y.values[[1]])
AUC_RF <- round(performance(prediction(pred_with_prob_rf, 
                                       test_set$TARGET),'auc')@y.values[[1]],3)




#pacchetto naive bayes

train_set<-as.data.frame(training_set)
test_set<-as.data.frame(test_set)


#alleno NaiveBayes
bn <- naive.bayes(train_set, "TARGET")
fit_bn <- bn.fit(bn,train_set, method = "mle")

#alleno TreeAugmentedNetwork
tan <- tree.bayes(train_set, "TARGET")
fit_tan <- bn.fit(tan, train_set, method = "bayes")


#risultati naive bayes
pred <- predict(fit_bn, test_set)
confusionMatrix(pred,test_set[,1],positive='1')
recall(pred,test_set[,1],relevant = '1')
precision(pred,test_set[,1],relevant = '1')
F1_NB <- F1_Score(pred,test_set[,1],positive = '1')

# curva roc
ROC1 <- performance(prediction(as.numeric(pred), test_set$TARGET), 'tpr', 'fpr')
ROC_NB <- data.frame(x=ROC1@x.values[[1]], y=ROC1@y.values[[1]])
AUC_NB <- round(performance(prediction(as.numeric(pred),
                                       test_set$TARGET),'auc')@y.values[[1]],3)

# risultati tree augmented network
pred_tan <- predict(fit_tan, test_set,prob = TRUE)
confusionMatrix(pred_tan,test_set[,1],positive='1')
recall(pred_tan,test_set[,1],relevant = '1')
precision(pred_tan,test_set[,1],relevant = '1')
F1_TAN <- F1_Score(pred_tan,test_set[,1],positive = '1')
Accuracy(pred_tan,test_set[,1])

ROC2 <- performance(prediction(as.numeric(pred_tan), test_set$TARGET), 'tpr', 'fpr')
ROC_TAN <- data.frame(x=ROC2@x.values[[1]], y=ROC2@y.values[[1]])
AUC_TAN <- round(performance(prediction(as.numeric(pred_tan),
                                        test_set$TARGET),'auc')@y.values[[1]],3)



# grafico ROC
ggplot() + 
  
  geom_line(data = ROC_DT,aes(x,y,col="A"),show.legend = TRUE) +
  xlab('False Positive Rate') + ylab('True Positive Rate') +
  geom_line(data=ROC_RF,aes(x,y,col="B"),show.legend = TRUE) +
  geom_line(data = ROC_NB,aes(x,y,col="C"),show.legend = TRUE) +
  geom_line(data=ROC_TAN,aes(x,y,col="D"),show.legend = TRUE) +
  
  scale_colour_manual(name = "Model", values = c("A"="red","B"="blue", "C" = "dark green", "D" = "orange"),
                      labels = c("DT","RF", "NB", "TAN")) +
  annotate("text", x=0.6, y=0, 
           label= paste0("AUC DT = ",AUC_DT),
           col="red") +
  annotate("text", x=0.6, y=0.05, 
           label= paste0("AUC RF = ",AUC_RF),
           col="blue") +
  annotate("text", x=0.9, y=0, 
           label= paste0("AUC NB = ",AUC_NB),
           col="dark green") +
  annotate("text", x=0.9, y=0.05, 
           label= paste0("AUC TAN = ",AUC_TAN),
           col="orange")


#-------------------------------------------------------------------------------------------------------------------------




#PREPARAZIONE INIZIALE DATASET "raw_7_tic.csv" PER RFM E CLUSTERING


library(rfm)


df <- read.csv2("raw_7_tic.csv", na.strings = c("NA", ""))  #importo il dataset sugli scontrini


#trasformo la colonna Data nel formato omonimo e la inserisco nel dataset
data <- as.Date(df$DATETIME)  
df <- cbind(df,data)
df <- df[-9]

head(df)


df <- df %>% arrange(desc(data)) #ordino il dataset in base alla data,il range è dal 01/05/2018 al 30/04/2019


#creazione del dataset RFM iniziale con monetary (somma delle spese per ogni cliente), frequency (espressa nel numero totale di
#prodotti acquistati per ogni cliente e nwl numero di transazioni di ogni cliente) e le date del primo e dell'ultimo acquisto
#sempre per cliente
df %>% group_by(ID_CLI) %>% summarise(data_primo_acquisto  = min(data) ,
                                      data_ultimo_acquisto = max(data) ,
                                      spesa_totale = sum(IMPORTO_LORDO),
                                      frequency = n() ,
                                      numero_transazioni = n_distinct(ID_SCONTRINO)) -> df2


recency <- c((as.Date("2019-04-30")) - df2$data_ultimo_acquisto) #creazione colonna recency: distanza trascorsa dall'ultimo acquisto
recency
df2 <- cbind(df2, recency)
head(df2)



#sulla base della divisione dei valori in cinque quantili viene assegnato un punteggio da 1 a 5 

quantili <- quantile(df2$spesa_totale, probs = seq(0, 1, 0.20))

df2 %>% mutate(MonetaryScore = ifelse(spesa_totale <= quantili[2], 1 , 
                                      ifelse(spesa_totale > quantili[2] & spesa_totale <= quantili[3], 2 ,
                                             ifelse(spesa_totale > quantili[3] & spesa_totale <= quantili[4], 3 ,
                                                    ifelse(spesa_totale > quantili[4] & spesa_totale <= quantili[5], 4 , 5 ))))) -> dsRFM2


#per la  recency viene considerato maggiormente chi ha un punteggio basso rispetto ad uno alto (distanza minore dall'ultimo acquisto)

quantili <- quantile(dsRFM2$recency, probs = seq(0, 1, 0.20))

dsRFM2 %>% mutate(RecencyScore =  ifelse(recency <= quantili[2], 5 , 
                                         ifelse(recency > quantili[2] & recency <= quantili[3], 4 ,
                                                ifelse(recency > quantili[3] & recency <= quantili[4], 3 ,
                                                       ifelse(recency > quantili[4] & recency <= quantili[5], 2 , 1 ))))) -> dsRFM2


quantili <- quantile(dsRFM2$numero_transazioni, probs = seq(0, 1, 0.20))

dsRFM2 %>% mutate(FrequencyScore =  ifelse(numero_transazioni <= quantili[2], 1 , 
                                           ifelse(numero_transazioni > quantili[2] & numero_transazioni <= quantili[3], 2 ,
                                                  ifelse(numero_transazioni > quantili[3] & numero_transazioni <= quantili[4], 3 ,
                                                         ifelse(numero_transazioni > quantili[4] & numero_transazioni <= quantili[5], 4 , 5 ))))) -> dsRFM2


#lo scoring totale è dato dalla somma di quelli delle 3 categorie


dsRFM2 %>% mutate(TotalScore = MonetaryScore + RecencyScore + FrequencyScore) -> dsRFM2

dsRFM2 %>% mutate(TotalScoreMean = MonetaryScore/3 + RecencyScore/3 + FrequencyScore/3) -> dsRFM2


dsRFM2 %>% 
  rename(
    monetary = spesa_totale ,
    recency_score = RecencyScore,
    frequency_score =  FrequencyScore
  ) -> dfRFMprova2


#attraverso il pacchetto rfm di R ottengo lo stesso dataset finale ed è possibile ottenere alcuni grafici utili per osservare i dati

analysis_data <- lubridate::as_date('2019-05-01', tz = 'UTC')

rfm_result <- rfm_table_customer(dfRFMprova2, ID_CLI, numero_transazioni,recency,monetary,analysis_data, recency_bins = 5, frequency_bins = 5, monetary_bins = 5)

rfm_heatmap(rfm_result)

rfm_bar_chart(rfm_result)

rfm_rm_plot(rfm_result)

rfm_fm_plot(rfm_result)

rfm_rf_plot(rfm_result)

str(dfRFMprova2)
dfRFMprova2 <- mutate(dfRFMprova2, segment = "NA")

# Per ogni categoria --> A = customers che spendono di più; B = customers che spendone nella media; C = customers che spendono di meno

# Segmento clienti "inattivi"

dfRFMprova2$segment[which(dfRFMprova2$recency > 30*3)] = "inactive"

# Segmento clienti "cold"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary > 200)] = "cold high A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "cold high B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary < 50)] = "cold high C"


dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary > 200)] = "cold medium A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "cold medium B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary < 50)] = "cold medium C"


dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary > 200)] = "cold low A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "cold low B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*3 & dfRFMprova2$recency > 30*2 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary < 50)] = "cold low C"

# Segmento clienti "warm"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary > 200)] = "warm high A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "warm high B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary < 50)] = "warm high C"


dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary > 200)] = "warm medium A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "warm medium B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary < 50)] = "warm medium C"


dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary > 200)] = "warm low A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "warm low B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30*2 & dfRFMprova2$recency > 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary < 50)] = "warm low C"


# Segmento clienti "attivi"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary > 200)] = "active high A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "active high B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 3 &
                            dfRFMprova2$monetary < 50)] = "active high C"



dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary > 200)] = "active medium A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "active medium B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni > 1 & dfRFMprova2$numero_transazioni <= 3 &
                            dfRFMprova2$monetary < 50)] = "active medium C"



dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary > 200)] = "active low A"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary >= 50 & dfRFMprova2$monetary <= 200)] = "active low B"

dfRFMprova2$segment[which(dfRFMprova2$recency <= 30 &
                            dfRFMprova2$numero_transazioni == 1 &
                            dfRFMprova2$monetary < 50)] = "active low C"


#Osservo le percentuali dei segmenti ottenuti

prop.table(table(dfRFMprova2$segment))
dfRFMprova2$segment %>% table() %>% prop.table() %>% `*`(100) %>% round(2)

summary(dfRFMprova2)




#Attrverso alcuni grafici osservo le mediane delle 3 caratteristiche per ogni segmento

#Median Recency
cluster_data <-
  dfRFMprova2 %>%
  group_by(segment) %>%
  select(segment, recency) %>%
  summarize(median(recency)) %>%
  rename(segment = segment, avg_recency = `median(recency)`) %>%
  arrange(avg_recency)




library(RColorBrewer)


ggplot(cluster_data, aes(segment, avg_recency)) +
  geom_bar(stat = "identity") +
  xlab("Segment") + ylab("Median Recency") +
  ggtitle("Mean Recency by Segment") +
  coord_flip() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

#Median Frequency

cluster_data1 <-
  dfRFMprova2 %>%
  group_by(segment) %>%
  select(segment, frequency) %>%
  summarize(median(frequency)) %>%
  rename(segment = segment, avg_frequency = `median(frequency)`) %>%
  arrange(avg_frequency)



ggplot(cluster_data1, aes(segment, avg_frequency)) +
  geom_bar(stat = "identity") +
  xlab("Segment") + ylab("Median Frequency") +
  ggtitle("Mean Frequency by Segment") +
  coord_flip() +
  theme(plot.title = element_text(hjust = 0.5)
  )

#Median Monetary Value
cluster_data2 <-
  dfRFMprova2 %>%
  group_by(segment) %>%
  select(segment, numero_transazioni) %>%
  summarize(median(numero_transazioni)) %>%
  rename(segment = segment, avg_monetary = `median(numero_transazioni)`) %>%
  arrange(avg_monetary)



ggplot(cluster_data2, aes(segment, avg_monetary)) +
  geom_bar(stat = "identity") +
  xlab("Segment") + ylab("Median Monetary Value") +
  ggtitle("Mean Monetary by Segment") +
  coord_flip() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )
