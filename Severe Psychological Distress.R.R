library("caret")
library(pROC)
library(ROSE)
library(ggplot2)
library(epiDisplay)
library(h2o)
h2o.init()
##真实数据
datt<-read.csv("C:/Users/tiantian/Desktop/data_original.csv",head=T)
datt$Y <-as.factor(datt$Y)
datt=na.omit(datt)
datt=datt[,-59]
#datt$Y[which(datt$Y==1)]<-2
#datt$Y[which(datt$Y==0)]<-1

for (i in 2:54){
  datt[,i]<-as.integer(datt[,i])
}


#t值检验
chiall<-NULL
tall<-NULL
for(i in 2:53){
  datt2<-NULL
  datt2<-cbind(datt[,1],datt[,i])
  t<-NULL
  t<-t.test(datt2[,2]~datt2[,1],var.equal=T)
  tall[i]<-t$p.value
}
tall

#卡方检验
for(i in 54:58){
  datt3<-NULL
  datt3<-cbind(datt[,1],datt[,i])
  chi<-NULL
  chi<-chisq.test(datt3[,2],datt3[,1])
  chiall[i]<-chi$p.value
}
chiall

###样本平衡后的数据
dat<- SMOTE(datt[,-1],dat[,1],dup_size = 0,K=5)
write.csv(datt,"C:/Users/tiantian/Desktop/data_smote.csv")
dat<-read.csv("C:/Users/tiantian/Desktop/data_smote.csv",head=T)
dat<-dat[,-1]
dat$Y <-as.factor(dat$Y)
dat=na.omit(dat)

for (i in 3:54){
  dat[,i]<-as.integer(dat[,i])
}

##########GLM model
grid<-NULL
glm0 <- h2o.grid("glm",
                 y=1,
                 training_frame = as.h2o(dat),
                 nfolds=5,
                 alpha = 0.00005,
                 seed = 1)
glmlist <- h2o.getGrid(glm0@grid_id, sort_by="auc", decreasing=T)
glmlist
glm_b <- h2o.getModel(glmlist@model_ids[[1]])
glm_b #alpha=0.00005  0.802877
#summary(glm_b)

importance_glm=h2o.varimp(glm_b)
write.table(importance_glm,"C:/Users/tiantian/Desktop/importance_glm.csv",row.names=FALSE,col.names=TRUE,sep=",")
importance_glm
x_glm = as.vector(importance_glm[1:20,"variable"])
grid<-NULL
glm = h2o.grid("glm", training_frame = as.h2o(dat), x=x_glm, y=1, 
               nfolds = 5, alpha = 0.00005,seed=1)
glmgrid = h2o.getGrid(glm@grid_id,sort_by = "auc",decreasing = T)
glmgrid
glm_f=h2o.getModel(glmgrid@model_ids[[1]])
glm_f 
###ROC-GLM
perf <- h2o.performance(glm_f ,as.h2o(datt))
plot(perf, type = "roc",)
pred <- h2o.predict(object = glm_f , newdata =as.h2o(datt))
pred
pred_glm<-cbind(unlist(as.data.frame(pred$p1)),datt$Y)
roc1<- roc(pred_glm[,2],pred_glm[,1])
auc(roc1)#0.7676
ggroc(roc1)

#####RF
rf0 <- h2o.grid("randomForest", y=1,
                training_frame = as.h2o(dat),
                nfolds=5,
                binomial_double_trees = T,
                hyper_params = list(ntrees = 300,
                                    mtries = 7,
                                    max_depth = 30),seed = 1)
rflist <- h2o.getGrid(rf0@grid_id, sort_by="auc", decreasing=T)
rflist
rf_b <- h2o.getModel(rf0@model_ids[[1]])
rf_b#max_depth 30， mtries 7，ntrees 300，accuracy 0.936345
importance_rf=h2o.varimp(rf_b)
importance_rf
x_rf = as.vector(importance_rf[1:20,"variable"])
x_rf
write.table(importance_rf,"C:/Users/tiantian/Desktop/importance_rf.csv",row.names=FALSE,col.names=TRUE,sep=",")

grid<-NULL
rf_0 <- h2o.grid("randomForest", x=x_rf,y=1, training_frame = as.h2o(dat), 
                 nfolds = 5, binomial_double_trees = T,
                 hyper_params = list(ntrees = 300,
                                     mtries = 7,
                                     max_depth = 30),seed = 1)
rfgrid = h2o.getGrid(rf_0@grid_id,sort_by = "auc",decreasing = T)
rfgrid
rf_f=h2o.getModel(rfgrid@model_ids[[1]])
rf_f #ntrees = 200,mtries = 3,max_depth = 10 0.88998
perf2 <- h2o.performance(rf_f , as.h2o(datt))
plot(perf2, type = "roc",)
pred2 <- h2o.predict(object = rf_f , newdata = as.h2o(datt))
pred2
pred_rf<-cbind(unlist(as.data.frame(pred2$p1)),datt$Y)
typeof(pred_rf)
###ROC-GLM
roc2<- roc(pred_rf[,2],pred_rf[,1])
auc(roc2)#0.9598
ggroc(roc2)

###DL
set.seed(1)
dl14 = h2o.deeplearning(y=1, training_frame = as.h2o(dat), hidden=c(200,100), activation="Maxout", nfolds=5, seed=1)
summary(dl14) 
importance_dl=h2o.varimp(dl14)
write.table(importance_dl,"C:/Users/tiantian/Desktop/importance_dl.csv",row.names=FALSE,col.names=TRUE,sep=",")
x_dl = as.vector(importance_dl[1:20,"variable"])
x_dl
set.seed(1)
dl_12 = h2o.deeplearning(x=x_dl,y=1, training_frame = as.h2o(dat), hidden=c(200,100),activation="Tanh",nfolds=5, seed=1)
summary(dl_12) 
perf3 <- h2o.performance(dl_12 , as.h2o(datt))
plot(perf3, type = "roc",)
pred3 <- h2o.predict(object = dl_12 , newdata = as.h2o(datt))
pred3
pred_dl<-cbind(unlist(as.data.frame(pred3$p1)),datt$Y)
typeof(pred_dl)
###ROC-GLM
roc3<- roc(pred_dl[,2],pred_dl[,1])
auc(roc3)#0.7840

###GBM
grid<-NULL

gbm_grid1 <- h2o.grid("gbm", y = 1,
                      grid_id = "gbm_grid1",
                      training_frame =as.h2o(dat),
                      seed = 1,
                      nfolds=5,
                      hyper_params = list(learn_rate = 0.10,
                                          max_depth =10,
                                          ntrees=500))
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid1",
                             sort_by = "auc",
                             decreasing = TRUE)
best_gbm1 <- h2o.getModel(gbm_gridperf1@model_ids[[1]])
importance_gbm<-h2o.varimp(best_gbm1)
x_gbm = as.vector(importance_rf[1:20,"variable"])
grid<-NULL
gbm_grid0 <- h2o.grid("gbm", x=x_gbm,y = 1,
                      grid_id = "gbm_grid0",
                      training_frame = as.h2o(dat),
                      seed = 1,
                      nfolds=5,
                      hyper_params = list(learn_rate = 0.10,
                                          max_depth =10,
                                          ntrees=500))
gbm_gridperf0 <- h2o.getGrid(grid_id = "gbm_grid0",
                             sort_by = "auc",
                             decreasing = TRUE)
best_gbm0 <- h2o.getModel(gbm_gridperf0@model_ids[[1]])
perf4 <- h2o.performance(best_gbm0 , as.h2o(datt))
plot(perf4, type = "roc")
pred4 <- h2o.predict(object = best_gbm0 , newdata = as.h2o(datt))
pred4
pred_gbm<-cbind(unlist(as.data.frame(pred4$p1)),datt$Y)
typeof(pred_gbm)
###ROC-GBM
roc4<- roc(pred_gbm[,2],pred_gbm[,1])
auc(roc4)#0.9744
ggroc(roc4)


####LR模型
lm<-glm(Y~.,data=dat,family=binomial)
a<-lm$coefficients
logit.for<-step(lm,direction="forward",set.seed=1)
summary(logit.for)
#OR计算
ORDF1<-logistic.display(logit.for)
ORDF1

prob<-predict(object =logit.for,newdata=datt,type = "response")
pred<-ifelse(prob>=0.5,"yes","no")
roc5 <- roc(datt$Y,prob)
auc(roc5)#0.7699

###作图
g10<-ggroc(list(GLM=roc1, RandomForest=roc2, 
                DL= roc3, GBM=roc4, LR=roc5))+
  scale_colour_manual(values=c("#F17B77", "#81B41B", "#10BABD","#C27EF1","#501802FF"))+
  theme_classic()+
  annotate("text", x = 0.6, y = 0.65, label = "AUC:0.7676",col = "#F17B77")+
  annotate("text", x = 0.85, y = 0.8 ,label = "AUC: 0.9598",col = "#81B41B")+
  annotate("text", x = 0.82, y = 0.7, label = "AUC:0.7840",col = "#10BABD")+
  annotate("text", x = 0.95, y = 0.98, label = "AUC:0.9744",col = "#C27EF1")+
  annotate("text", x = 0.75, y = 0.5, label = "AUC:0.7699",col = "#501802FF")+
  ggtitle("Machine Learning")+ 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.ticks.length=unit(-0.1, "cm"), 
        panel.border = element_rect(fill=c(0,0,0,1),color="black", size=0.5, linetype="solid"),
        axis.text.x = element_text(margin=unit(c(0.15,0.15,0.15,0.15), "cm")), 
        axis.text.y = element_text(margin=unit(c(0.15,0.15,0.15,0.15), "cm")),aspect.ratio = 1,legend.position = c(.8, .3),legend.background = element_rect(fill = "white",colour = "black"))
g10













