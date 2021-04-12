library(readxl)
output2000 <- read_excel("output2000.xlsx")

ggplot(output, aes(x=Model, y=F1, fill=factor(`Sample Size`))) + geom_boxplot() + ggtitle("Boxplot of F1 Score by Model Type and Sample Size") +
  theme(plot.title = element_text(hjust = 0.5)) + xlab("Model Type") + ylab("F1 Score") + labs(fill = "Sample Size of Test")
ggplot(output, aes(x=Model, y=`AUC`, fill=factor(`Sample Size`))) + geom_boxplot() + ggtitle("Boxplot of AUC Score by Model Type and Sample Size") +
  theme(plot.title = element_text(hjust = 0.5)) + xlab("Model Type") + ylab("AUC Score") + labs(fill = "Sample Size of Test")
