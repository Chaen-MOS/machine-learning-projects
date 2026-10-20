#  Part 1: Data Loading and Exploratory Data Analysis

data_file = file.path(
  "data",
  "spambase.data"
)

if (!file.exists(data_file)) {
  stop(
    "spambase.data was not found. ",
    "Place it inside the data folder."
  )
}

feature_names = c(
  "word_freq_make",
  "word_freq_address",
  "word_freq_all",
  "word_freq_3d",
  "word_freq_our",
  "word_freq_over",
  "word_freq_remove",
  "word_freq_internet",
  "word_freq_order",
  "word_freq_mail",
  "word_freq_receive",
  "word_freq_will",
  "word_freq_people",
  "word_freq_report",
  "word_freq_addresses",
  "word_freq_free",
  "word_freq_business",
  "word_freq_email",
  "word_freq_you",
  "word_freq_credit",
  "word_freq_your",
  "word_freq_font",
  "word_freq_000",
  "word_freq_money",
  "word_freq_hp",
  "word_freq_hpl",
  "word_freq_george",
  "word_freq_650",
  "word_freq_lab",
  "word_freq_labs",
  "word_freq_telnet",
  "word_freq_857",
  "word_freq_data",
  "word_freq_415",
  "word_freq_85",
  "word_freq_technology",
  "word_freq_1999",
  "word_freq_parts",
  "word_freq_pm",
  "word_freq_direct",
  "word_freq_cs",
  "word_freq_meeting",
  "word_freq_original",
  "word_freq_project",
  "word_freq_re",
  "word_freq_edu",
  "word_freq_table",
  "word_freq_conference",
  "char_freq_semicolon",
  "char_freq_left_paren",
  "char_freq_left_bracket",
  "char_freq_exclamation",
  "char_freq_dollar",
  "char_freq_hash",
  "capital_run_length_average",
  "capital_run_length_longest",
  "capital_run_length_total"
)

names(spam) = c(feature_names,"Spam_Status")

# Check the data
dim(spam)
names(spam)
summary(spam)
sapply(spam,class)

# Check missing values
sum(is.na(spam))

# Check the response
table(spam$Spam_Status)

# All 57 predictors are continuous numeric features
numeric_cols = 1:57


# 1.1 Histogram (numeric data)
hist(spam$word_freq_free,
     main = "Distribution of word_freq_free",
     xlab = "word_freq_free",
     col = "steelblue",
     breaks = 20)

hist(spam$capital_run_length_total,
     main = "Distribution of Capital Run Length Total",
     xlab = "capital_run_length_total",
     col = "lightgreen",
     breaks = 20)


# 1.2 Barplot (response class)
barplot(table(spam$Spam_Status),
        main = "Frequency of Spam Status",
        xlab = "Spam Status (0 = Non-Spam, 1 = Spam)",
        ylab = "Count",
        col = "lightblue")


# 1.3 Scatter plot (numeric vs numeric data)
plot(spam$word_freq_free, spam$word_freq_money,
     main = "word_freq_money vs word_freq_free",
     xlab = "word_freq_free",
     ylab = "word_freq_money",
     col = "blue", pch = 19)


# 1.4 Box plot (numeric vs response)
boxplot(word_freq_free ~ Spam_Status, data=spam,
        main = "word_freq_free by Spam Status",
        xlab = "Spam Status", ylab = "word_freq_free",
        col = "lightyellow")

boxplot(capital_run_length_total ~ Spam_Status, data=spam,
        main = "Capital Run Length Total by Spam Status",
        xlab = "Spam Status", ylab = "capital_run_length_total",
        col = "lightyellow")


# 1.5 Pair plot (multivariate numeric data)
pair_vars = c("word_freq_free",
              "word_freq_money",
              "word_freq_your",
              "char_freq_exclamation",
              "capital_run_length_total")

pairs(spam[,pair_vars],
      main = "Pairwise Relationships Among Key Numeric Variables")


# 1.6 Additional univariate analysis for all numeric features
par(mfrow=c(3,3))
for (column in numeric_cols) {
  hist(spam[,column], main=names(spam)[column], xlab="")
}
par(mfrow=c(1,1))


# 1.7 Boxplot analysis for all numeric features
par(mfrow=c(3,3))
for (column in numeric_cols) {
  boxplot(spam[,column], main=names(spam)[column], ylab="")
}
par(mfrow=c(1,1))


# 1.8 Feature-response analysis for numeric features
summary(spam[spam$Spam_Status==0,numeric_cols])
summary(spam[spam$Spam_Status==1,numeric_cols])

par(mfrow=c(3,3))
for (column in numeric_cols) {
  boxplot(spam[,column] ~ spam$Spam_Status,
          main=names(spam)[column],
          xlab="Spam Status",
          ylab="")
}
par(mfrow=c(1,1))


# 1.9 Correlation analysis
cor(spam[,numeric_cols])


#  Part 2: Unsupervised Learning ： PCA
# PCA is exploratory in this assignment and is not used as a
# supervised model input. If PCA is used for modelling later,
# fit it on training data only.
spam.pca = spam[,numeric_cols]
dim(spam.pca)
features = colnames(spam.pca);features

# Identify the center of the data
apply(spam.pca,2,mean)
colMeans(spam.pca)

# Find the variance of the columns
apply(spam.pca,2,var)

# standardization
pca = prcomp(spam.pca,scale=TRUE);pca
summary(pca)

screeplot(pca)
screeplot(pca,type="line")
names(pca)
pca$sdev
pca$rotation
pca$center
pca$scale
pca$x

# A full biplot
biplot(pca,scale=0)

cc = c("black","red")
plot(pca$x[,1:2],
     col=cc[as.integer(as.factor(spam$Spam_Status))],
     pch=16,
     xlab="PC1",
     ylab="PC2",
     main="First and Second PCs")

# Proportion of Variance Explained
the.eigenvalues = pca$sdev^2
PVE = the.eigenvalues/sum(the.eigenvalues)
cumsum(PVE)

par(mfrow=c(1,2))
plot(PVE,
     xlab="Principal Component",
     ylab="Proportion of Variance Explained (PVE)",
     ylim=c(0,1),
     type="b",
     col="blue")

plot(cumsum(PVE),
     xlab="Principal Component",
     ylab="Cumulative Proportion of Variance Explained (CPVE)",
     ylim=c(0,1),
     type="b",
     col="red")
par(mfrow=c(1,1))



#  Performance Measurements for Classification Problem
performance = function(xtab, desc=""){
    cat(desc,"\n")
    print(xtab)

    ACR = sum(diag(xtab))/sum(xtab)
    CI  = binom.test(sum(diag(xtab)), sum(xtab))$conf.int
    cat("\n        Accuracy :", ACR)
    cat("\n          95% CI : (", CI[1], ",", CI[2], ")\n")

    if(nrow(xtab)>2){
        n  = sum(xtab)
        ni = apply(xtab,1,sum)
        nj = apply(xtab,2,sum)
        p0 = sum(diag(xtab))/n
        pc = sum(ni*nj)/n^2
        Kappa = (p0-pc)/(1-pc)
        cat("\n           Kappa :", Kappa, "\n")
        cat("\nStatistics by Class:\n")
        lvls = dimnames(xtab)[[2]]
        sensitivity = c()
        specificity = c()
        ppv = c()
        npv = c()

        for(i in 1:length(lvls)){
            sensitivity[i] = xtab[i,i]/sum(xtab[,i])
            specificity[i] = sum(xtab[-i,-i])/sum(xtab[,-i])
            ppv[i] = xtab[i,i]/sum(xtab[i,])
            npv[i] = sum(xtab[-i,-i])/sum(xtab[-i,])
        }

        b = data.frame(rbind(sensitivity,specificity,ppv,npv))
        names(b) = lvls
        print(b)

    } else {
         TPR = xtab[1,1]/sum(xtab[,1])
         TNR = xtab[2,2]/sum(xtab[,2])
         PPV = xtab[1,1]/sum(xtab[1,])
         NPV = xtab[2,2]/sum(xtab[2,])
         FPR = 1-TNR
         FNR = 1-TPR

         RandomAccuracy = (sum(xtab[,2])*sum(xtab[2,]) +
           sum(xtab[,1])*sum(xtab[1,]))/(sum(xtab)^2)

         Kappa = (ACR-RandomAccuracy)/(1-RandomAccuracy)

         cat("\n           Kappa :", Kappa, "\n")
         cat("\n     Sensitivity :", TPR)
         cat("\n     Specificity :", TNR)
         cat("\n  Pos Pred Value :", PPV)
         cat("\n  Neg Pred Value :", NPV)
         cat("\n             FPR :", FPR)
         cat("\n             FNR :", FNR, "\n")
         cat("\n'Positive' Class :", dimnames(xtab)[[1]][1], "\n")
    }
}



#  Part 3: Data Preparation for Supervised Learning
spam.fac = spam
spam.fac$Spam_Status = factor(spam.fac$Spam_Status)
sapply(spam.fac,class)

set.seed(123)

spam_tag1 = spam.fac[spam.fac$Spam_Status=="1",]
spam_tag0 = spam.fac[spam.fac$Spam_Status=="0",]

split_indices = function(n,
                         train_prop=0.6,
                         validation_prop=0.1) {

    shuffled_idx = sample(seq_len(n))

    train_size = round(train_prop*n)
    validation_size = round(validation_prop*n)

    list(
      train=shuffled_idx[1:train_size],
      validation=shuffled_idx[
        (train_size+1):(train_size+validation_size)
      ],
      test=shuffled_idx[
        (train_size+validation_size+1):n
      ]
    )
}

tag1_split = split_indices(nrow(spam_tag1))
tag0_split = split_indices(nrow(spam_tag0))

spam.train = rbind(
  spam_tag1[tag1_split$train,],
  spam_tag0[tag0_split$train,]
)

spam.validation = rbind(
  spam_tag1[tag1_split$validation,],
  spam_tag0[tag0_split$validation,]
)

spam.test = rbind(
  spam_tag1[tag1_split$test,],
  spam_tag0[tag0_split$test,]
)

summary(spam.validation)
summary(spam.test)
table(spam.train$Spam_Status)
table(spam.validation$Spam_Status)
table(spam.test$Spam_Status)


#  Part 4: Supervised Learning Model 1: Logistic Regression
# Logistic regression uses the data without normalization
logreg_m = glm(Spam_Status~.,data=spam.train,family=binomial)
print(summary(logreg_m))
print(anova(logreg_m,test="Chisq"))

# Perform binary classification using conditional probability
spam.test.prob = predict(logreg_m,newdata=spam.test,type='response')

yhat = ifelse(spam.test.prob >= 0.5,"1","0")

# Reorder the confusion matrix so Spam = 1 is the Positive class
cfmat.logreg = table(
  factor(yhat,levels=c("1","0")),
  factor(spam.test$Spam_Status,levels=c("1","0"))
)

performance(cfmat.logreg,
            "Performance of the Full Logistic Regression Model")

logreg_accuracy = sum(diag(cfmat.logreg))/sum(cfmat.logreg)
logreg_sensitivity = cfmat.logreg[1,1]/sum(cfmat.logreg[,1])
logreg_specificity = cfmat.logreg[2,2]/sum(cfmat.logreg[,2])
logreg_FPR = 1-logreg_specificity
logreg_FNR = 1-logreg_sensitivity

logreg_RandomAccuracy =
  (sum(cfmat.logreg[,2])*sum(cfmat.logreg[2,]) +
     sum(cfmat.logreg[,1])*sum(cfmat.logreg[1,]))/
  (sum(cfmat.logreg)^2)

logreg_Kappa =
  (logreg_accuracy-logreg_RandomAccuracy)/
  (1-logreg_RandomAccuracy)

# Validation performance is used for model comparison.
spam.validation.prob = predict(
  logreg_m,
  newdata=spam.validation,
  type='response'
)

yhat.validation = ifelse(
  spam.validation.prob >= 0.5,
  "1",
  "0"
)

cfmat.logreg.validation = table(
  factor(yhat.validation,levels=c("1","0")),
  factor(spam.validation$Spam_Status,levels=c("1","0"))
)

performance(
  cfmat.logreg.validation,
  "Validation performance of the Full Logistic Regression Model"
)

logreg_validation_accuracy =
  sum(diag(cfmat.logreg.validation))/sum(cfmat.logreg.validation)

logreg_validation_sensitivity =
  cfmat.logreg.validation[1,1]/
  sum(cfmat.logreg.validation[,1])

logreg_validation_specificity =
  cfmat.logreg.validation[2,2]/
  sum(cfmat.logreg.validation[,2])

# Reduced Logistic Regression
# Retain predictors with p-value < 0.05 from the full Logistic Regression model

reduced.logreg_m = glm(
  Spam_Status~
    word_freq_our+
    word_freq_over+
    word_freq_remove+
    word_freq_internet+
    word_freq_mail+
    word_freq_free+
    word_freq_business+
    word_freq_your+
    word_freq_000+
    word_freq_hp+
    word_freq_hpl+
    word_freq_george+
    word_freq_technology+
    word_freq_meeting+
    word_freq_project+
    word_freq_re+
    word_freq_edu+
    word_freq_conference+
    char_freq_semicolon+
    char_freq_exclamation+
    char_freq_dollar+
    capital_run_length_average+
    capital_run_length_longest,
  data=spam.train,
  family=binomial
)

print(summary(reduced.logreg_m))

# Compare reduced and full logistic regression models
print(anova(reduced.logreg_m,logreg_m,test="Chisq"))

# Validate the reduced logistic regression model
spam.test.prob.reduced = predict(
  reduced.logreg_m,
  newdata=spam.test,
  type='response'
)

yhat.reduced = ifelse(spam.test.prob.reduced >= 0.5,"1","0")

cfmat.logreg.reduced = table(
  factor(yhat.reduced,levels=c("1","0")),
  factor(spam.test$Spam_Status,levels=c("1","0"))
)

performance(cfmat.logreg.reduced,
            "Performance of the Reduced Logistic Regression Model")

reduced_logreg_accuracy = sum(diag(cfmat.logreg.reduced))/sum(cfmat.logreg.reduced)
reduced_logreg_sensitivity = cfmat.logreg.reduced[1,1]/sum(cfmat.logreg.reduced[,1])
reduced_logreg_specificity = cfmat.logreg.reduced[2,2]/sum(cfmat.logreg.reduced[,2])
reduced_logreg_FPR = 1-reduced_logreg_specificity
reduced_logreg_FNR = 1-reduced_logreg_sensitivity

reduced_logreg_RandomAccuracy =
  (sum(cfmat.logreg.reduced[,2])*sum(cfmat.logreg.reduced[2,]) +
     sum(cfmat.logreg.reduced[,1])*sum(cfmat.logreg.reduced[1,]))/
  (sum(cfmat.logreg.reduced)^2)

reduced_logreg_Kappa =
  (reduced_logreg_accuracy-reduced_logreg_RandomAccuracy)/
  (1-reduced_logreg_RandomAccuracy)


# Validate the reduced model on the validation set
spam.validation.prob.reduced = predict(
  reduced.logreg_m,
  newdata=spam.validation,
  type='response'
)

yhat.validation.reduced = ifelse(
  spam.validation.prob.reduced >= 0.5,
  "1",
  "0"
)

cfmat.logreg.reduced.validation = table(
  factor(yhat.validation.reduced,levels=c("1","0")),
  factor(spam.validation$Spam_Status,levels=c("1","0"))
)

performance(
  cfmat.logreg.reduced.validation,
  "Validation performance of the Reduced Logistic Regression Model"
)

reduced_logreg_validation_accuracy =
  sum(diag(cfmat.logreg.reduced.validation))/
  sum(cfmat.logreg.reduced.validation)

reduced_logreg_validation_sensitivity =
  cfmat.logreg.reduced.validation[1,1]/
  sum(cfmat.logreg.reduced.validation[,1])

reduced_logreg_validation_specificity =
  cfmat.logreg.reduced.validation[2,2]/
  sum(cfmat.logreg.reduced.validation[,2])

#  Part 5: Supervised Learning Model 2: Decision Tree
library(tree)

# Train the classification tree model
tree.spam = tree(Spam_Status~.,spam.train)

print(tree.spam)
summary(tree.spam)

# Diagram representation
plot(tree.spam)
text(tree.spam,cex=0.6)

# Validate the trained CART tree model
tree.pred = predict(tree.spam,spam.test,type="class")

cfmat.tree = table(
  factor(tree.pred,levels=c("1","0")),
  factor(spam.test$Spam_Status,levels=c("1","0"))
)

performance(cfmat.tree,"\nSpambase with tree::tree")


# Validate the unpruned tree on the validation set
tree.validation.pred = predict(
  tree.spam,
  spam.validation,
  type="class"
)

cfmat.tree.validation = table(
  factor(tree.validation.pred,levels=c("1","0")),
  factor(spam.validation$Spam_Status,levels=c("1","0"))
)

performance(
  cfmat.tree.validation,
  "\nValidation performance of tree::tree"
)

# Pruning the trained tree model
set.seed(5)

spam_tree_cv = cv.tree(tree.spam,FUN=prune.misclass)

par(mfrow=c(1,2))
plot(spam_tree_cv)

plot(spam_tree_cv$size,
     spam_tree_cv$dev/nrow(spam.train),
     type="b",
     xlab="Tree Size",
     ylab="CV Misclassification Rate")

# Find tree size with minimum cross-validation error
min_idx = max(which(spam_tree_cv$dev==min(spam_tree_cv$dev)))
num_terminal_nodes = spam_tree_cv$size[min_idx]

prune.spam = prune.misclass(
  tree.spam,
  best=num_terminal_nodes
)

# Compare pruned tree and original tree
par(mfrow=c(1,2))

plot(prune.spam)
text(prune.spam,cex=0.6)
title("Pruned Tree")

plot(tree.spam)
text(tree.spam,cex=0.6)
title("Original Tree")

prune.pred = predict(
  prune.spam,
  spam.test,
  type="class"
)

cfmat.tree.prune = table(
  factor(prune.pred,levels=c("1","0")),
  factor(spam.test$Spam_Status,levels=c("1","0"))
)

performance(cfmat.tree.prune,
            "\nPruned tree::tree to minimal nodes")

par(mfrow=c(1,1))

# Validate the pruned tree on the validation set
prune.validation.pred = predict(
  prune.spam,
  spam.validation,
  type="class"
)

cfmat.tree.prune.validation = table(
  factor(prune.validation.pred,levels=c("1","0")),
  factor(spam.validation$Spam_Status,levels=c("1","0"))
)

performance(
  cfmat.tree.prune.validation,
  "\nValidation performance of pruned tree::tree"
)

tree_validation_accuracy =
  sum(diag(cfmat.tree.prune.validation))/
  sum(cfmat.tree.prune.validation)

tree_validation_sensitivity =
  cfmat.tree.prune.validation[1,1]/
  sum(cfmat.tree.prune.validation[,1])

tree_validation_specificity =
  cfmat.tree.prune.validation[2,2]/
  sum(cfmat.tree.prune.validation[,2])

tree_accuracy = sum(diag(cfmat.tree.prune))/sum(cfmat.tree.prune)
tree_sensitivity = cfmat.tree.prune[1,1]/sum(cfmat.tree.prune[,1])
tree_specificity = cfmat.tree.prune[2,2]/sum(cfmat.tree.prune[,2])
tree_FPR = 1-tree_specificity
tree_FNR = 1-tree_sensitivity

tree_RandomAccuracy =
  (sum(cfmat.tree.prune[,2])*sum(cfmat.tree.prune[2,]) +
     sum(cfmat.tree.prune[,1])*sum(cfmat.tree.prune[1,]))/
  (sum(cfmat.tree.prune)^2)

tree_Kappa =
  (tree_accuracy-tree_RandomAccuracy)/
  (1-tree_RandomAccuracy)


#  Part 6: Supervised Learning Model 3: kNN
# Use the same stratified sampling indices with original numeric data
spam_tag1.numeric = spam[spam$Spam_Status==1,]
spam_tag0.numeric = spam[spam$Spam_Status==0,]

spam.train.numeric = rbind(
  spam_tag1.numeric[tag1_split$train,],
  spam_tag0.numeric[tag0_split$train,]
)

spam.validation.numeric = rbind(
  spam_tag1.numeric[tag1_split$validation,],
  spam_tag0.numeric[tag0_split$validation,]
)

spam.test.numeric = rbind(
  spam_tag1.numeric[tag1_split$test,],
  spam_tag0.numeric[tag0_split$test,]
)

summary(spam.validation.numeric)
summary(spam.test.numeric)


#  Feature Scaling with Standardization
cat("\nData Preparation/Preprocessing: Data standardisation ...\n")

normalise.vec <- function(column,ref.col) {
    return ((column-mean(ref.col))/sd(ref.col))
}

spam.train.knn = spam.train.numeric
spam.validation.knn = spam.validation.numeric
spam.test.knn = spam.test.numeric

for(column in numeric_cols){
    spam.train.knn[,column] = normalise.vec(
      spam.train.knn[,column],
      spam.train.numeric[,column]
    )

    spam.validation.knn[,column] = normalise.vec(
      spam.validation.knn[,column],
      spam.train.numeric[,column]
    )

    spam.test.knn[,column] = normalise.vec(
      spam.test.knn[,column],
      spam.train.numeric[,column]
    )
}



#  Modelling with kNN & Evaluation for Binary Classification
library(class)

cat("\nTraining and validation with kNN ...\n\n")

# Select k using validation data only.
knn_validation_results = data.frame(
  k=c(3,5,10),
  Accuracy=NA,
  Sensitivity=NA,
  Specificity=NA
)

for(i in 1:nrow(knn_validation_results)) {

  current_k = knn_validation_results$k[i]

  validation_prediction = knn(
    spam.train.knn[,1:57],
    spam.validation.knn[,1:57],
    factor(spam.train.knn[,58]),
    k=current_k
  )

  validation_table = table(
    factor(validation_prediction,levels=c("1","0")),
    factor(spam.validation.knn$Spam_Status,
           levels=c("1","0"))
  )

  knn_validation_results$Accuracy[i] =
    sum(diag(validation_table))/sum(validation_table)

  knn_validation_results$Sensitivity[i] =
    validation_table[1,1]/sum(validation_table[,1])

  knn_validation_results$Specificity[i] =
    validation_table[2,2]/sum(validation_table[,2])
}

print("--- kNN Validation Results ---")
print(knn_validation_results)

best_knn_k = knn_validation_results$k[
  which.max(knn_validation_results$Accuracy)
]

cat(
  "\nSelected k using validation accuracy:",
  best_knn_k,
  "\n"
)

# k = 3
set.seed(123)

yhat.k3 = knn(
  spam.train.knn[,1:57],
  spam.test.knn[,1:57],
  factor(spam.train.knn[,58]),
  k=3
)

cftable.k3 = table(
  factor(yhat.k3,levels=c("1","0")),
  factor(spam.test.knn$Spam_Status,levels=c("1","0"))
)

performance(cftable.k3,
            "Confusion matrix and performance of kNN (k=3)")

knn3_accuracy = sum(diag(cftable.k3))/sum(cftable.k3)
knn3_sensitivity = cftable.k3[1,1]/sum(cftable.k3[,1])
knn3_specificity = cftable.k3[2,2]/sum(cftable.k3[,2])
knn3_FPR = 1-knn3_specificity
knn3_FNR = 1-knn3_sensitivity
knn3_RandomAccuracy =
  (sum(cftable.k3[,2])*sum(cftable.k3[2,]) +
     sum(cftable.k3[,1])*sum(cftable.k3[1,]))/
  (sum(cftable.k3)^2)
knn3_Kappa =
  (knn3_accuracy-knn3_RandomAccuracy)/(1-knn3_RandomAccuracy)


# k = 5
set.seed(123)

yhat.k5 = knn(
  spam.train.knn[,1:57],
  spam.test.knn[,1:57],
  factor(spam.train.knn[,58]),
  k=5
)

cftable.k5 = table(
  factor(yhat.k5,levels=c("1","0")),
  factor(spam.test.knn$Spam_Status,levels=c("1","0"))
)

performance(cftable.k5,
            "Confusion matrix and performance of kNN (k=5)")

knn5_accuracy = sum(diag(cftable.k5))/sum(cftable.k5)
knn5_sensitivity = cftable.k5[1,1]/sum(cftable.k5[,1])
knn5_specificity = cftable.k5[2,2]/sum(cftable.k5[,2])
knn5_FPR = 1-knn5_specificity
knn5_FNR = 1-knn5_sensitivity
knn5_RandomAccuracy =
  (sum(cftable.k5[,2])*sum(cftable.k5[2,]) +
     sum(cftable.k5[,1])*sum(cftable.k5[1,]))/
  (sum(cftable.k5)^2)
knn5_Kappa =
  (knn5_accuracy-knn5_RandomAccuracy)/(1-knn5_RandomAccuracy)


# k = 10
set.seed(123)

yhat.k10 = knn(
  spam.train.knn[,1:57],
  spam.test.knn[,1:57],
  factor(spam.train.knn[,58]),
  k=10
)

cftable.k10 = table(
  factor(yhat.k10,levels=c("1","0")),
  factor(spam.test.knn$Spam_Status,levels=c("1","0"))
)

performance(cftable.k10,
            "Confusion matrix and performance of kNN (k=10)")

knn10_accuracy = sum(diag(cftable.k10))/sum(cftable.k10)
knn10_sensitivity = cftable.k10[1,1]/sum(cftable.k10[,1])
knn10_specificity = cftable.k10[2,2]/sum(cftable.k10[,2])
knn10_FPR = 1-knn10_specificity
knn10_FNR = 1-knn10_sensitivity
knn10_RandomAccuracy =
  (sum(cftable.k10[,2])*sum(cftable.k10[2,]) +
     sum(cftable.k10[,1])*sum(cftable.k10[1,]))/
  (sum(cftable.k10)^2)
knn10_Kappa =
  (knn10_accuracy-knn10_RandomAccuracy)/(1-knn10_RandomAccuracy)


#  Part 7: Model Evaluation & Performance Comparison
# 7.1 kNN Performance Comparison

knn_performance_table = data.frame(
  Model=c(
    "Full kNN",
    "Full kNN",
    "Full kNN"
  ),
  k=c(3,5,10),
  Accuracy=c(
    knn3_accuracy,
    knn5_accuracy,
    knn10_accuracy
  ),
  Sensitivity=c(
    knn3_sensitivity,
    knn5_sensitivity,
    knn10_sensitivity
  ),
  Specificity=c(
    knn3_specificity,
    knn5_specificity,
    knn10_specificity
  ),
  FPR=c(
    knn3_FPR,
    knn5_FPR,
    knn10_FPR
  ),
  FNR=c(
    knn3_FNR,
    knn5_FNR,
    knn10_FNR
  ),
  Kappa=c(
    knn3_Kappa,
    knn5_Kappa,
    knn10_Kappa
  )
)

print("--- kNN Performance Comparison ---")
print(knn_performance_table)


# 7.2 Overall Model Performance Comparison
# These test results are reported for the final comparison only.
# They are not used to select the best model.
performance_table = data.frame(
  Model=c(
    "Full Logistic Regression",
    "Reduced Logistic Regression",
    "Pruned Decision Tree",
    "kNN (k=3)",
    "kNN (k=5)",
    "kNN (k=10)"
  ),
  Accuracy=c(
    logreg_accuracy,
    reduced_logreg_accuracy,
    tree_accuracy,
    knn3_accuracy,
    knn5_accuracy,
    knn10_accuracy
  ),
  Sensitivity=c(
    logreg_sensitivity,
    reduced_logreg_sensitivity,
    tree_sensitivity,
    knn3_sensitivity,
    knn5_sensitivity,
    knn10_sensitivity
  ),
  Specificity=c(
    logreg_specificity,
    reduced_logreg_specificity,
    tree_specificity,
    knn3_specificity,
    knn5_specificity,
    knn10_specificity
  ),
  FPR=c(
    logreg_FPR,
    reduced_logreg_FPR,
    tree_FPR,
    knn3_FPR,
    knn5_FPR,
    knn10_FPR
  ),
  FNR=c(
    logreg_FNR,
    reduced_logreg_FNR,
    tree_FNR,
    knn3_FNR,
    knn5_FNR,
    knn10_FNR
  ),
  Kappa=c(
    logreg_Kappa,
    reduced_logreg_Kappa,
    tree_Kappa,
    knn3_Kappa,
    knn5_Kappa,
    knn10_Kappa
  )
)

print("--- Overall Model Performance Comparison ---")
print(performance_table)


# 7.3 Model selection using validation results
validation_performance_table = data.frame(
  Model=c(
    "Full Logistic Regression",
    "Reduced Logistic Regression",
    "Pruned Decision Tree",
    paste("kNN (k=", best_knn_k, ")", sep="")
  ),
  Accuracy=c(
    logreg_validation_accuracy,
    reduced_logreg_validation_accuracy,
    tree_validation_accuracy,
    knn_validation_results$Accuracy[
      knn_validation_results$k==best_knn_k
    ]
  ),
  Sensitivity=c(
    logreg_validation_sensitivity,
    reduced_logreg_validation_sensitivity,
    tree_validation_sensitivity,
    knn_validation_results$Sensitivity[
      knn_validation_results$k==best_knn_k
    ]
  ),
  Specificity=c(
    logreg_validation_specificity,
    reduced_logreg_validation_specificity,
    tree_validation_specificity,
    knn_validation_results$Specificity[
      knn_validation_results$k==best_knn_k
    ]
  )
)

print("--- Validation Model Selection Table ---")
print(validation_performance_table)

best_model_name =
  validation_performance_table$Model[
    which.max(validation_performance_table$Accuracy)
  ]

cat("\nBest Predictive Model based on validation Accuracy:",
    best_model_name,"\n")

best_sensitivity_name =
  validation_performance_table$Model[
    which.max(validation_performance_table$Sensitivity)
  ]

cat("Model with Highest validation Sensitivity for Spam Email Detection:",
    best_sensitivity_name,"\n")

cat(
  "\nThe test-set comparison above is reported after model selection ",
  "and is not used to choose the best model.\n"
)

