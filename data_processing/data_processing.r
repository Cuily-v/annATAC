rm(list=ls())
load(file = "./scacatlist.Rdata") # Use Read10X_h5 to read the generated .h5 file and save it as data in the form of .Rdata.
ls()
all_data <- list()
for(i in 1:5){
	data = scataclist[[i]]@assays$RNA@layers$counts
	rownames(data) <- rownames(as.matrix(scataclist[[i]]@assays$RNA@features))
	colnames(data) <- rownames(as.matrix(scataclist[[i]]@assays$RNA@cells))
	all_data[[i]] <- data
}

for(index in 1:5){
	chr_result <- data.frame()
	data<-as.matrix(all_data[[index]])	
	chromosome = rownames(data)
	a <- gsub("chrX", "chr23", chromosome)	
	chromosome <- gsub("chrY", "chr24", a)		
	chromosome <- as.matrix(chromosome)

	for (x in 1:24){
	  chri = grep(paste('^chr',x,':',sep=""),chromosome)
	  chri_data=data[chri,]
	  class <- data.frame()
	  for (j in 1:dim(chri_data)[1]){
	    rownames(chri_data)[j] <- strsplit(rownames(chri_data),':')[[j]][2] 
		a = as.numeric((strsplit(rownames(chri_data)[j],'-')[1])[[1]][1])%/%5000
		class <- append(class,a)
	  }
	  class<-as.matrix(class)
	  chri_data = cbind(chri_data,class)
	  colnam <- colnames(chri_data)
	  colnam <- as.matrix(colnam)	
	  chri_data = t(matrix(unlist(chri_data), ncol= dim(chri_data)[1] , byrow= TRUE ))		  
	  chri_data<-ceiling(aggregate(chri_data, by=list(chri_data[,dim(chri_data)[2]]),mean))
	  chri_data$Group.1 <- paste('chr_',x,'_',chri_data$Group.1,sep="")   
	  chri_data <- chri_data[,-dim(chri_data)[2]]	  
	  colnames(chri_data) <- c("Group.1",colnam[-dim(chri_data)[2],])
	  chr_result <- rbind(chr_result,as.data.frame(chri_data))	  
	}	
}

