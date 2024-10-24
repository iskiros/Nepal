## @knitr Rain_Corr_MC_Variable_isotopes


################################### FUNCTION DEFINITION #############################################
# - One of Ed’s rain correction softwares
# - Good to figure out how to use Monte Carlo in future




#ETT 17/11/2021  Significant update
# Based on rain_corr_MC_variable_isotopes_rev2 used in Katy's paper
#Previous update:  now uses a uniform distribution to create the range in the rain data.  Previously used a mixture between SW and average dilute river
#Previous update: Added in GNIP rain weighted data for runif sampling of this range
#The S and O isotope data are now supplied prior to this script.

#THIS SCRIPT: 
#3) MC on all river cation and anion, rain cation and anion and stable isotope data
#4) Corrects cation, anion and stable isotope data for rain
#5) Stores everything in a df called MC.


#DEPENDENCIES:
#source("Libraries.R")
#source("Plotly_setup.R")
#source("Data_wrangling.R") #Prepare all the I-S data
#source("Data_wrangling_Mek.R") #Prepare all the Mek data
#source("Prep.Rain.Corr-I-S.R")#Add in the S and O isotope data for the rain, including GNIP
#source("Prep.Rain.Corr-M.R")#Add in the S and O isotope data for the rain, including GNIP
#source("Bind_I.S.M.R") #put the I-S and Mek tibbles together as one







################################################################################
# Monte-Carlo FAST ETT -----------------------------------------------
# This generates MC data on river cation and anions (2.5% 1sig error), rain cations and anions, (10% 1sig error) and isotopes (assuming only the instrumental 1sig errors) 
#Note the supppress warnings massively increases the speed of this since it is not writing anything to screen
{
{tic()
#remove(MC)
#remove(Monte_Carlo_Array)#Clear this to avoid any mishaps and to save RAM
# MC to do errors Prep the data 

  
  Monte_Carlo_Array<-Master.data%>%slice(rep(1:n(), each = n_simulations))
  #Define the uncertainties on the rain cation and anion data
  err_water<-0.025 #5% 2SD
  err_rain<-0.1  #20%2SD
  

  
  #THEN DO THE MC.
  #Instead of using a loop to fill Monte_Carlo_array with data, it uses sapply.  This should be faster, but is still slow.    Seems to work fine.  sapply works over specific cols
  
  
  #WATER CATIONS and ANIONS
  colnames(Monte_Carlo_Array)
  selected_cols<-c("Ca","K","Mg","Na","S","Si","Sr","Cl","SO4-IC","HCO3")
  suppressWarnings(Monte_Carlo_Array[,c(selected_cols)]<-apply(Monte_Carlo_Array[,c(selected_cols)], c(1,2),function(x) rnorm(1,mean=x,sd=err_water*x)))
  #Monte_Carlo_Array[,c("Ca.umolL"),]
  
  #RAIN CATIONS and ANIONS
  selected_cols<-c("Ca.rain","Na.rain","Mg.rain","K.rain","S.rain","Si.rain","Sr.rain","SO4-IC.rain","Cl.rain","HCO3.rain")
  #suppressWarnings(Monte_Carlo_Array[,selected_cols]<-sapply(Monte_Carlo_Array[,selected_cols],function(x)
  #rnorm(1,mean=as.numeric(as.character(x)),sd=err_rain*as.numeric(as.character(x)))))
  suppressWarnings(Monte_Carlo_Array[,c(selected_cols)]<-apply(Monte_Carlo_Array[,c(selected_cols)], c(1,2),function(x) rnorm(1,mean=x,sd=err_rain*x)))
  #Monte_Carlo_Array[,c("RainCa_umolL"),]
  #colnames(Monte_Carlo_Array)
  
  #ISOTOPES
  #Define a function for rnorm, so that this can easily use sample specific errors.  Is this the right thing to do?
  #Should we actually be similulating more than the analytical error?
  fun1 <- function(x,y) {
    z <- rnorm(1,mean=as.numeric(as.character(x)), sd=as.numeric(as.character(y)))
    return(z)
  } 
  
  fun2 <- function(x,y) {
    z <- runif(1, min=x, max=y)
    return(z)
  } 
  
  #Then use mapply 
  #Convert 2SD d18O error to 1SD THIS SHOULD MOVE TO GLOBAL PREP
  suppressWarnings(Monte_Carlo_Array[,c("2SD.d18O")]<-0.5*as.numeric(as.character(Monte_Carlo_Array[,c("2SD.d18O")])))
  suppressWarnings(Monte_Carlo_Array[,c("d18O")]<-mapply(fun1,  Monte_Carlo_Array[,c("d18O")],Monte_Carlo_Array[,c("2SD.d18O")]))#Note it's now 1SD
  suppressWarnings(Monte_Carlo_Array[,c("d18O_GNIP")]<-mapply(fun2,  Monte_Carlo_Array[,c("perc25_d18O")],Monte_Carlo_Array[,c("perc75_d18O")]))
  suppressWarnings(Monte_Carlo_Array[,c("d18O_GNIP_10")]<-mapply(fun2,  Monte_Carlo_Array[,c("perc10_d18O")],Monte_Carlo_Array[,c("perc75_d18O")]))
  suppressWarnings(Monte_Carlo_Array[,c("d18O-SO4")]<-mapply(fun1,  Monte_Carlo_Array[,c("d18O-SO4")],0.5*Monte_Carlo_Array[,c("2SD.d18O-SO4")]))
  suppressWarnings(Monte_Carlo_Array[,c("d34S-SO4")]<-mapply(fun1,  Monte_Carlo_Array[,c("d34S-SO4")],0.5*Monte_Carlo_Array[,c("2SD.d34S-SO4")]))
  suppressWarnings(Monte_Carlo_Array[,c("d18O-SO4.rain")]<-mapply(fun2,  Monte_Carlo_Array[,c("d18O_SO4_rain_min")],Monte_Carlo_Array[,c("d18O_SO4_rain_max")]))
  suppressWarnings(Monte_Carlo_Array[,c("d34S-SO4.rain")]<-mapply(fun2,  Monte_Carlo_Array[,c("d34S_rain_min")],Monte_Carlo_Array[,c("d34S_rain_max")]))
  
    
  
  
  
  
  #Monte_Carlo_Array[,c("RainCa_umolL"),]
  
  MC<-Monte_Carlo_Array
  remove(Monte_Carlo_Array)  
  
  toc()      
  
  #subset(MC, !is.na(d18O_SO4_H2O))  
  

}

################################################################################
# Calculcate rain ratios -----------------------------------------------


MC<-MC%>%mutate(
  RainCa_Cl = Ca.rain / Cl.rain, 
  RainK_Cl = K.rain / Cl.rain,
  RainMg_Cl = Mg.rain / Cl.rain,
  RainNa_Cl = Na.rain / Cl.rain,
  RainSi_Cl = Si.rain / Cl.rain,
  RainSr_Cl = Sr.rain / Cl.rain,
  RainSO4_Cl = `SO4-IC.rain` / Cl.rain,
  RainS_Cl = S.rain / Cl.rain,
  RainHCO3_Cl = HCO3.rain / Cl.rain)
  
  
  
  



################################################################################
# Correct River Cl -----------------------------------------------
#The Cl- correction is given by:
#Cl*river = Clriver - Clrain
{
MC$`Cl*` <- MC$Cl - MC$Cl.rain #This varies with location depending on closest rain sample
#Ensure that Clstar is not negative, if it is negative, assume Clstar = 0 which is assuming that all the CL in the river is from rain
MC$`Cl*` <- ifelse(MC$`Cl*`>0, MC$`Cl*`, 0) 
#MekongWaterRainCorr[MekongWaterRainCorr$`Cl*.umolL`<0 ,c("SampleID","River", "SampleType", "Cl*.umolL", "Cl.umolL", "Symbol")]#ETT Not sure what this line is doing?
MC[, c("WaterID", "Cl", "Cl*", "Cl.rain")]
}

################################################################################
# Correct other cations anions and isotopes-----------------------------------------------

#Correct river water using the local rain
#Set values to zero if <0

MC<-MC%>%mutate(
# K*river
`K*` =K - ((Cl - `Cl*`) * RainK_Cl),
`K*`=ifelse(`K*`<0, 0, `K*`),
# Na*river
`Na*` = Na - ((Cl - `Cl*`) * RainNa_Cl),
`Na*`=ifelse(`Na*`<0, 0, `Na*`),
# Ca*river
`Ca*` = Ca - ((Cl - `Cl*`) * RainCa_Cl),
`Ca*`=ifelse(`Ca*`<0, 0, `Ca*`),
# Mg*river
`Mg*` = Mg - ((Cl - `Cl*`) * RainMg_Cl),
`Mg*`=ifelse(`Mg*`<0, 0, `Mg*`),
# Mg*river
`SO4*` = `SO4-IC` - ((Cl - `Cl*`) * RainSO4_Cl),
`SO4*`=ifelse(`SO4*`<0, 0, `SO4*`),
# S*river
`S*` = S- ((Cl - `Cl*`) * RainS_Cl),
`S*`=ifelse(`S*`<0, 0, `S*`),
# Si*river
`Si*` = Si- ((Cl - `Cl*`) * RainSi_Cl),
`Si*`=ifelse(`Si*`<0, 0, `Si*`),
# Sr*river
`Sr*` = Sr- ((Cl - `Cl*`) * RainSr_Cl),
`Sr*`=ifelse(`Sr*`<0, 0, `Sr*`),
# HCO3*river
`HCO3*` = HCO3- ((Cl - `Cl*`) * RainHCO3_Cl),
`HCO3*`=ifelse(`HCO3*`<0, 0, `HCO3*`),
#Correct the isotopes
`d34S-SO4*`=(`d34S-SO4`*`SO4-IC`-`d34S-SO4.rain`*(`SO4-IC`-`SO4*`))/`SO4*`,
`d18O-SO4*`=(`d18O-SO4`*`SO4-IC`-`d18O-SO4.rain`*(`SO4-IC`-`SO4*`))/`SO4*`

)
#View(MC)

}#Enclose entire script
#RAIN CORRECTED DATA STORED IN DF CALLED MC TO PASS ON TO OTHER SCRIPTS FOR PLOTTING ETC

#saveRDS(MC, file="MC_rain_corr.rds")

# Perform calculations on MC data
# Example calculation: Calculate the mean and standard deviation of RainCa_Cl
mean_RainCa_Cl <- mean(MC$RainCa_Cl)
sd_RainCa_Cl <- sd(MC$RainCa_Cl)

# Print the results
print(mean_RainCa_Cl)
print(sd_RainCa_Cl)