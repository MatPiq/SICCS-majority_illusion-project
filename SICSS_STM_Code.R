###############################################################################
# Title     : Cleaned code for STM analysis of Reddit
# Author(s) : Jeongho Choi
# Date      : July 3rd, 2023
###############################################################################

library(dplyr)
library(tm)
library(stringr)
library(quanteda)
library(stm)
library(tibbletime)
library(rio)
library(reshape2)
library(ggplot2)

  # Note: All codes are drawn from Roberts et al (2019). stm: An R Packgage for Structural Topic Models
  #       I highly recommend read this article and follow their codes. I found them extremely helpful.

## 2.1 Import data =============================================================
getwd()
setwd("C:/Users/helxi/Documents/GitHub/SICSS-project")

speeches <-import("thread_texts.csv")     #Thread-centric
speeches2<-import("user_texts.csv")       #User-centric
speeches3<-import("user_valet_texts.csv") #User-centric (filltered by Valet)
colnames(speeches)
colnames(speeches2)
colnames(speeches3)
head(speeches)
head(speeches2)
head(speeches3)

## 2.2 Pre-Processing ==========================================================
  #STM package has a function called TextProcessor to do all text processing at once.
  #You can tweack the code to customize text processing for your research (see ?textProcessor)
  #But, for running the stm later, you can use other text processing methods like we learned in SCISS
  
  # Thread-centric model
    processed <- textProcessor(speeches$text, metadata = speeches, language = "swedish")
  
    out <- prepDocuments(processed$documents, processed$vocab,
                         processed$meta)
    #create corpus
    docs<-out$documents
    vocab<-out$vocab
    meta<-out$meta
  
    colnames(meta)
  
  # User-centric model
    processed2 <- textProcessor(speeches2$text, metadata = speeches2, language = "swedish")
    out2 <- prepDocuments(processed2$documents, processed2$vocab,
                         processed2$meta)
    #create corpus
    docs2<-out2$documents
    vocab2<-out2$vocab
    meta2<-out2$meta
    colnames(meta2)
  
  # User-centric model with valet(?) text data
    processed3 <- textProcessor(speeches3$text, metadata = speeches3, language = "swedish")
    out3 <- prepDocuments(processed3$documents, processed3$vocab,
                          processed3$meta)
    #create corpus
    docs3 <-out3$documents
    vocab3<-out3$vocab
    meta3 <-out3$meta
    colnames(meta3)
    
## 2.2 Search K ================================================================
  # STM package can compare the performance of models with different K.
  # For more detail, please refer to stm:R Package for Structural TopicModels
  storage <- searchK(docs3,  # document 
                     vocab3, # Vocab
                     K=c(5,10,15,20,25,30,35,40,45,50), # K can be changed
                     prevalence =~ user,                # meta-data (user-name in this example)
                     max.em.its = 75,
                     gamma.prior = "L1", #default is "Pooled", but we use L1 because of the model convergence
                     data = meta3, init.type = "Spectral", seed=9999, verbose =TRUE)
  print(storage$results)
  options(repr.plot.width=6, repr.plot.height=6)

  #Figure 1
  plot(storage)
  
  #exclusivity = words in topic 1 cannot be found in topic 2 (higher better)
  #semantic coherence = words in topic 1 should co-occur within the same document (higher better)
  #held-out = how well does the model predict the held-out data (higher better)
    # Code book: With the built-in function make.heldout we construct a dataset in which 10% of documents
    # have half of their words removed. We can then evaluate the quality of inferred topics on the same evaluation set across all models.
  #residuals = just like OLS (lower better)
  
  # The most important criteria would be residuals.
  # If residuals are not really different, you can move on to exclusivity and semantic coherence
  # Held-out likelihood and its lower-bound are not that important
  
  # Compare semantic coherence and exclusivity
  model20<-stm(documents=out3$documents, 
               vocab=out3$vocab, 
               prevalence =~ user,  
               K=20, gamma.prior = "L1",
               data=out3$meta, 
               init.type = "Spectral", verbose=TRUE, seed=9999, max.em.its = 75)
  model25<-stm(documents=out3$documents, 
               vocab=out3$vocab, 
               prevalence =~ user,  
               K=25, gamma.prior = "L1",
               data=out3$meta, 
               init.type = "Spectral", verbose=TRUE, seed=9999, max.em.its = 75)
  model30<-stm(documents=out3$documents, 
               vocab=out3$vocab, 
               prevalence =~ user,  
               K=30, gamma.prior = "L1",
               data=out3$meta, 
               init.type = "Spectral", verbose=TRUE, seed=9999, max.em.its = 75)
  
  M10ExSem<-as.data.frame(cbind(c(1:20), 
                                exclusivity(model20), 
                                semanticCoherence(model=model20, docs3),
                                "Mod20")) 
  M15ExSem<-as.data.frame(cbind(c(1:25), 
                                exclusivity(model25), 
                                semanticCoherence(model=model25, docs3), 
                                "Mod25")) 
  M20ExSem<-as.data.frame(cbind(c(1:30), 
                                exclusivity(model30), 
                                semanticCoherence(model=model30, docs3), 
                                "Mod30"))
  
  ModsExSem<-rbind(M10ExSem, M15ExSem, M20ExSem)
  colnames(ModsExSem)<-c("K","Exclusivity", "SemanticCoherence", "Model")
  
  ModsExSem$Exclusivity<-as.numeric(as.character(ModsExSem$Exclusivity))
  ModsExSem$SemanticCoherence<-as.numeric(as.character(ModsExSem$SemanticCoherence))
  
  options(repr.plot.width=7, repr.plot.height=7, repr.plot.res=100)
  
  plotexcoer<-ggplot(ModsExSem, aes(SemanticCoherence, Exclusivity, color = Model))+geom_point(size = 2, alpha = 0.7) + 
    geom_text(aes(label=K), nudge_x=.05, nudge_y=.05)+
    labs(x = "Semantic coherence",
         y = "Exclusivity",
         title = "Comparing exclusivity and semantic coherence")
  
  #Figure 2
  plotexcoer
  
  # However, you should do qualitative comparison to see which models produce better topics
  # Run quantitative comparison first, and find a few number of K which look working well.
  # Then, run the models for selected K and qualitatively compare the result
  
## 2.3 Running STM ==============================
  #FREX : weights words by their overall frequency and how exclusive they are to the topic.
  #Lift: weights words by dividing by their frequency in other topics, therefore giving higher weight to words that appear less frequently in other topics.
  #Score: divides the log frequency of the world in the topic by the log frequency of the word in other topics.
  #So, basically, all these three measures are to find words that are more likely to appear in a topic and less likely to appear in other topics. But they do so in a different way.
  
  model30Prrateby<-stm(documents=out3$documents, 
                       vocab=out3$vocab, 
                       prevalence =~ user,  
                       K=20, gamma.prior = "L1", #default is "Pooled", but we use L1 because of the model convergence
                       data=out3$meta, 
                       init.type = "Spectral", verbose=TRUE, seed=9999, max.em.its = 75)
  
## 2.4 Visualization ==================================================================
  #Topic results
  labelTopics(model30Prrateby)
  
  # TOPIC distribution for each individual
  valet_user_centric_dist<-model30Prrateby$theta
  valet_user_centric_dist<-data.frame(valet_user_centric_dist)
  valet_user_centric_dist$user<-meta3$user
  
  # Draw topic proportion of one user
  new.df<-melt(valet_user_centric_dist,id.vars="user")
  colnames(new.df)
  new.df<-new.df[order(new.df$user, new.df$variable),]
  names(new.df)=c("user","topics","value")  
  
  user1<-new.df %>% filter(user == "6XJPCmTMB7gm3rMhUKE5")
  
  blue_palette <- colorRampPalette(colors = c("lightblue", "darkblue"))(20)
  ggplot(user1, aes(x=topics,y=value, fill=topics))+geom_histogram(stat="identity")+
    scale_fill_manual(values = blue_palette)
  
  #Global distribution
  plot.STM(model20Prrateby,type="summary", labeltype = c("prob"))
  plot.STM(model20Prrateby,type="summary", labeltype = c("frex"), n=5)
  plot.STM(model30Prrateby,type="summary", labeltype = c("frex"), n=5)
  
  #Label version
  plot.STM(model30Prrateby,type="summary", labeltype = c("frex"), n=1,
           main = "Expected Topic Proportions over the Whole Corpus",
           xlim = c(0, 0.2),
           topic.names = c("Topic 1: Election issues/ problems",
                           "Topic 2: Deleted commnets", 
                           "Topic 3: Energy and Climate",
                           "Topic 4: Coalition formation",
                           "Topic 5: Right-wing democratic critique",
                           "Topic 6: Voting strategies/ system", 
                           "Topic 7: Discussing extremism",
                           "Topic 8: Anti-LGBTQI/ Anti-immigration",
                           "Topic 9: Anti-establishment", 
                           "Topic 10: Bashing moderates", 
                           "Topic 11: Miscellaneous", 
                           "Topic 12: NATO",
                           "Topic 13: Malmo politics",
                           "Topic 14: Right-wing critique",
                           "Topic 15: Ukraine invasion",
                           "Topic 16: Sex work", 
                           "Topic 17: Military threat from Russia",
                           "Topic 18: Liberal party politics - Values",
                           "Topic 19: Drug policy", 
                           "Topic 20: Liberal party politics - Taxes and Money"),
           custom.labels = c(""))
  
  #Word version
  plot.STM(model30Prrateby,type="summary", labeltype = c("frex"), n=4,
           main = "Expected Topic Proportions over the Whole Corpus",
           xlim = c(0, 0.2))
  
  #Topic correlation
  mod.out.corr<-topicCorr(model20Prrateby, cutoff = 0.1) # Cutoff is 0.1
  plot(mod.out.corr)

  #Sample Text (Go into the Appendix)
  thoughts1 <- findThoughts(model30Prrateby, texts = meta3$text, 
                            topics = 11, # Which topics?
                            n = 3)$docs[[1]] #You can change n=? to decide how many text to read
  thoughts1  
 