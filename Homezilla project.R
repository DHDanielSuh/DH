library(tidyverse)
library(readxl)

properties<- read_xlsx("W16604-XLS-ENG.xlsx")
browsing<- read_xlsx("W16604-XLS-ENG.xlsx",sheet = "Browsing Data")

summary(browsing)
summary(properties)



a2 <- properties %>% 
  group_by(subtype, bedrooms) %>% 
  summarize(supply=n_distinct(`Web ID`))

a2 %>% 
  ggplot(aes(x=subtype, y=supply, 
             fill = as.factor(bedrooms))) +
  geom_bar(stat = "identity",position="dodge", 
           width = 0.6,col="black")

a2 %>% ggplot(aes(x=bedrooms, y=avg_price)) +
  geom_bar(stat = "identity", 
           position="dodge", 
           width = 0.6,col="black")

a2$bedrooms <- as.factor(a2$bedrooms)


a2 %>% ggplot(aes(x=bedrooms, y=supply, fill = subtype))+
  geom_bar(stat = "identity",position="dodge", width = 0.6,col="black")+scale_x_discrete(breaks=seq(0,10,1))

a2 %>% ggplot(aes(x=bedrooms, y=avg_price, fill = subtype))+
  geom_bar(stat = "identity",position="dodge", width = 0.6,col="black")+scale_x_discrete(breaks=seq(0,10,1))



a2%>% ggplot(aes(x=subtype, y=number_of_customers, fill = as.factor(bedrooms)))+
  geom_bar(stat = "identity",position="dodge", width = 0.6,col="black")


photos_subtype <- properties %>% 
  group_by(`Photo ID`,`Web ID`,subtype) %>% 
  summarise(n_distinct(`PhotoID))

View(properties)                                                          
a3<-properties %>% 
  group_by(subtype, bedrooms) %>% 
  summarize(supply=n_distinct(`Web ID`), Numofph = sum(Num_of_photos))

c<-browsing%>%
  group_by(`Web ID`)%>% 
  summarize(Num_customers=n_distinct(a=`Customer ID`),
            Num_of_photos=n_distinct(`Photo ID`))
properties<-merge(properties, c)
View(c)

a2<-properties %>% group_by(subtype, bedrooms)%>% summarize(supply=n_distinct(`Web ID`))
a3<-properties %>% group_by(subtype, bedrooms)%>% summarize(supply=n_distinct(`Web ID`), Numofph = sum(Num_of_photos))
a4.1 <- group_by(browsing, `Web ID`, `Customer ID`) %>% 
  summarise(Numofph = n_distinct(`Photo ID`),
            Totaltime = sum(`Time Viewed`))
# aaa<-browsing %>% unique(browsing$`Photo ID`)
#   group_by(`Web ID`,`Photo ID`) %>% 
#   select(`Web ID`,`Photo Tag 1`) 
# 
# View(aaa)

View(table(browsing$`Photo Tag 1`))

d <- browsing %>% group_by(`Web ID`, `Photo Tag 1`) %>% summarise(n1 = n_distinct(`Photo ID`))
d5 <- spread(d, `Photo Tag 1`, n1)
d5.1 <- merge(properties, d5)


View(d)
View(d5)
View(d5.1)


a5 <- d5.1 %>% 
  group_by(subtype, bedrooms) %>% 
  summarise(Avg_photo_ex = mean(exterior), 
            Avg_photo_in = mean(interior), 
            Avg_photo_floor = mean(floor))

View(a5)

a6 <- d5.1 %>% 
  group_by(subtype, bedrooms) %>% 
  summarise(Tot_photo_ex = sum(exterior), 
            Tot_photo_in = sum(interior), 
            Tot_photo_floor = sum(floor))

View(a6)

a6 %>% ggplot(aes(x = subtype, 
                  y = Tot_photo_ex, 
                  fill = as.factor(bedrooms))) + 
  geom_bar(position = position_dodge(), stat = "identity") + 
  geom_text(aes(label=Tot_photo_ex), vjust=1.1, color = "white", position = position_dodge(0.9), size = 5.5) + 
  scale_fill_brewer(palette = "Paired", name = "Number of Bedrooms") + 
  labs(title = "Total Exterior Photo",
       x = "Subtype",
       y = "Total Exterior Photo") + 
  theme(legend.position = "right")


a6 %>% ggplot(aes(x = subtype, 
                  y = Tot_photo_in, 
                  fill = as.factor(bedrooms))) + 
                geom_bar(position = position_dodge(), stat = "identity") +
                geom_text(aes(label=Tot_photo_in), vjust=1.1, color = "white", position = position_dodge(0.9), size = 5.5) + 
                scale_fill_brewer(palette = "Paired", name = "Number of Bedrooms") + 
  labs(title = "Total Interior Photo",
       x = "Subtype",
       y = "Total Interior Photo") + 
  theme(legend.position = "right")
              
서동현              

a4.1 <- group_by(browsing, `Web ID`, `Customer ID`) %>% 
  summarise(Numofph = n_distinct(`Photo ID`),
            Totaltime = sum(`Time Viewed`))

View(a4.1)
a5.4 <- merge(properties, a4.1)
View(a5.4)


d <- browsing %>% group_by(`Web ID`, `Photo Tag 1`) %>% summarise(n1 = n_distinct(`Photo ID`))
d5 <- spread(d, `Photo Tag 1`, n1)
d5.1 <- merge(properties, d5)

              

