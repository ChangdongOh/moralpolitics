library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(reshape)


mflist = c('care', 'fairness', 'ingroup', 'authority', 'purity')

dataprocess = function(filename, mf){
  data <- t(read.csv(paste0("result/wordvec/", filename, ".csv"), header = T, row.names = 1, check.names = F))
  data <- data[complete.cases(data),]
  
  me <- data[,str_detect(colnames(data), 'mean')]
  colnames(me) = mflist
  
  se <- data[,str_detect(colnames(data), 'se')]
  colnames(se) = mflist
  
  meandata = melt(me, id.vars = 'time')
  sedata = melt(se, id.vars = 'time')
  meanse = cbind(meandata, sedata$value)
  colnames(meanse) = c('time', 'mf', 'mean', 'se')
  
  meanse = meanse[meanse$mf == mf,]
  meanse = meanse[meanse$time != '2008-01|2008-02|2016-12', ]
  meanse$mf = str_replace_all(meanse$mf, mf, filename)
  
  return(meanse)
}

for(i in mflist){
  data = rbind(dataprocess('새누리', i), dataprocess('민주', i))
  colnames(data) = c('time', 'party', 'mean', 'se')
  
  ggplot(data, aes(x = time, y = mean, group = party, fill = party)) +
    geom_line() +
    geom_ribbon(aes(ymin = mean - se, ymax = mean + se), colour = 'black') +
    scale_x_discrete(breaks = c('2008-06', '2009-06', '2010-06', '2011-06',
                                '2012-06', '2013-06', '2014-06', '2015-06', '2016-06'),
                     label = c('2008', '2009', '2010', '2011',
                               '2012', '2013', '2014', '2015', '2016')) +    
    scale_fill_manual(values= c('#66CCFF', '#FF3366')) +
    theme(panel.grid.minor = element_blank(), 
          panel.grid.major = element_line(color = "gray50", size = 0.5), 
          panel.grid.major.x = element_blank(),
          panel.background = element_blank(),
          plot.title = element_text(hjust = 0.5),
          axis.ticks.length = unit(.25, "cm"),
          axis.text.x = element_text(size = 12, face = 'bold'),
          axis.ticks.y = element_blank(),
          axis.ticks.x = element_blank()) +
    labs(list(title = toupper(i),
              x = 'Period(Year)', 
              y = 'Cosine Similarity to the MFD words')) +
    guides(fill=guide_legend(title="Party"))
  
  ggsave(paste0('result/wordvec/', i, '.jpg'), width = 20, height = 10, units = 'cm')
}
