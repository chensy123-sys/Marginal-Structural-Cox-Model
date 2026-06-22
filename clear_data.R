library(haven); library(tibble); library(dplyr)
data <- read_xpt("data/illinois.xpt")
# data <- read.csv("illinois.csv")
attach(data)
head(data)
# CONTROL
# HIE
# JSIE
# HIE_A = HIE*ifelse(LAGREE==1, 1,0)
# JSIE_A = JSIE*ifelse(LAGREE==1, 1,0)
# table(HIE,HIE_A)
# table(JSIE,JSIE_A)

time <- REHIREDT-CLAIMDT
time[is.na(time)] <- 26*7
time2 <- REFILEDT-CLAIMDT
# hist(time2[time2>0])
time2[is.na(time2)] <- 26*7
hist(time[time>0])


LAGREE_new <- LAGREE
LAGREE_new[is.na(LAGREE_new)] <- 0

data_rec <- data %>% mutate(
  Z_HIE = HIE,
  Z_JSIE = JSIE,
  A_HIE = HIE*ifelse(LAGREE_new==1, 1,0),
  A_JSIE = JSIE*ifelse(LAGREE_new==1, 1,0),
  time_r = pmin(26*7,time),
  status_r = REHIREIN*((time<=26*7)*1),
  # time_f = time2,
  # status_f = REFILEIN
  ) %>%
  select(
    -HIE, -JSIE, # 两种干预措施
    -LAGREE, -CONTROL, # 是否同意研究，是否属于控制组
    # -CLAIMDT, # 个体进入时间
    # -BENYRBEG, # 失业救济金年度开始日期
    -REHIREDT,-REHIRE11,-REHIREIN, # 再就业指标
    -REFILEIN,-REFILEDT, # 再次申请失业
    -VCHPAID, # 是否领取转换津贴，可能是政策相应中介
    -WKPAID, -WKBENEFT, -WKSPDBYE, # 领取UI的后果，可能是中介变量
    -BENEPAID, -BENPDBYE, # 中介变量，领取了多少 UI 金额，受益年结束时间
    -BENYRBEG, -EXSTBEN1, -EXSTBENY, -RHIREARN
  ) %>%
  mutate(
    RACE = case_when(
      BLACK == 1     ~ "Black",
      WHITE == 1     ~ "White",
      HISPANIC == 1  ~ "Hispanic",
      NATVAMER == 1  ~ "NativeAmerican",
      OTHERACE == 1  ~ "Other",
    ),
    RACE = factor(RACE, levels = c("White", "Black", "Hispanic", "NativeAmerican", "Other"))
  ) %>%
  select(-BLACK,-WHITE,-HISPANIC,-NATVAMER,-OTHERACE) %>%
  # filter(!is.na(WGETOT6), !is.na(POSPEARN), !is.na(POSQEARN)) %>%
  filter(!is.na(AVPREARN), !is.na(PREPEARN)) %>%
  filter(time_r>0)
write.csv(data_rec,file = 'data/data_rec.csv')





# hist(data_rec$time_r)
# # hist(data_rec$time2-data_rec$time, breaks=100)

# table(data$REFILEIN, is.na(data$REFILEDT))


# plot(data$REFILEDT,data$REHIREDT)
# hist(data$REHIREDT-data$REFILEDT)


# # write.csv(data,file = 'illinois.csv')

# time <- REHIREDT-CLAIMDT
# # hist(time[time>0])
# status <- 1-is.na(time)
# time[is.na(time)] <- 26*7
# # hist(time[time>0])
# status <- status[which(time>0)]
# time <- time[which(time>0)]
# plot(survfit(Surv(time,status)~1))

# summary(coxph(Surv(time,status)~JSIE[which(time>0)] + HIE[which(time>0)]))


# # 各个人种
# table(BLACK+WHITE+HISPANIC+NATVAMER+OTHERACE)

# # 假设 df 是你的数据框
# binary_check <- sapply(data, function(col) n_distinct(col, na.rm = TRUE) == 2)

# # 查看哪些列是二元变量
# which(binary_check)






# table(LAGREE)
# table(CONTROL)


# LAGREE_new <- LAGREE +1
# LAGREE_new[is.na(LAGREE_new)] <- 0

# table(LAGREE_new, CONTROL*-1)




# LAGREE_new
# table(LAGREE_new, (JSIE==1)*1 + (HIE)*2)
