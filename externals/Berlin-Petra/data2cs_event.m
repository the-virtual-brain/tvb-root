function [cs,coh,nave]=data2cs_event(data,segleng,segshift,epleng,maxfreqbin,para);
% usage: [cs,coh,nave]=data2cs_event(data,segleng,segshift,epleng,maxfreqbin,para)
% 
% calculates cross-spectra and coherency from data for event-related measurement
% input: 
% data: ndat times nchan matrix each colum is the time-series in one
%             channel;
% segleng: length of each segment in bins, e.g. segleng=1000;  
% segshift: numer of bins by which neighboring segments are shifted;
%           e.g. segshift=segleng/2 makes overlapping segments
% epleng: length of each epoch
% maxfreqbin: max frequency in bins
% para: optional structure:
%       para.segave=0  -> no averaging across segments 
%       para.segave neq 0 -> averaging across segments (default is 1)% \
%       para.subave =1 subtracts the average across epochs,  
%       para.subave ~= 1 -> no subtraction (default is 1) 
%       IMPORTANT: if you just one epoch (e.g. for continuous data)
%         set para.subave=0 
%         
%       -> averaging across segments (default is 0)
%       para.proj must be a set of vector in channel space,  
%       if it exists then the output raw contains the single trial 
%       Fourier-transform in that channel   
%       para.zeropad=n  adds  n zeros at the end of each segment and at the end 
%                       of the window. default n=0 
%       para.mydetrend=1 (detrends linear trends in all segments; default=0 (no detrending))
%         
% output: 
% cs: nchan by chan by maxfreqbin by nseg tensor cs(:,:,f,i) contains 
%     the cross-spectrum at frequency f and segment i
% coh: complex coherency calculated from cs    
% nave: number of averages

subave=1; 

if nargin<6
    para=[];
end



segave=1;
mydetrend=0;
proj=[];
zeropad=0;
  if isfield(para,'segave')
    segave=para.segave;
  end 
   if isfield(para,'detrend')
    mydetrend=para.detrend;
  end 
  if isfield(para,'proj')
    proj=para.proj;
  end 
  if isfield(para,'subave')
    subave=para.subave;
  end 
 if isfield(para,'zeropad')
    zeropad=para.zeropad;
  end 
[ndum,npat]=size(proj);

[ndat,nchan]=size(data);
if npat>0 
   data=data*proj;
   nchan=npat;
end

maxfreqbin=min([maxfreqbin,floor((segleng+zeropad)/2)+1]);
nep=floor(ndat/epleng);

nseg=floor((epleng-segleng)/segshift)+1; %total number of segments



if segave==0
 cs=zeros(nchan,nchan,maxfreqbin,nseg); 
 av=zeros(nchan,maxfreqbin,nseg);
else
 cs=zeros(nchan,nchan,maxfreqbin); 
 av=zeros(nchan,maxfreqbin);
end

if npat>0
  if segave==0
    cs=zeros(nchan,nchan,maxfreqbin,nep,nseg); 
    av=zeros(nchan,maxfreqbin,nep,nseg);
  else
    cs=zeros(nchan,nchan,maxfreqbin,nep); 
    av=zeros(nchan,maxfreqbin,nep);
  end
end


mywindow=repmat(hanning(segleng),1,nchan);
if isfield(para,'mywindow');
    mywindow=repmat(para.mywindow,1,nchan);
end
   if zeropad>0
            mywindow=[mywindow;zeros(zeropad,nchan)];
   end
%figure;plot(mywindow);
nave=0;
for j=1:nep;
    dataep=data((j-1)*epleng+1:j*epleng,:);
    for i=1:nseg; %average over all segments;
        dataloc=dataep((i-1)*segshift+1:(i-1)*segshift+segleng,:);
         if mydetrend==1;
             dataloc=detrend(dataloc);
         end
         
        
        if zeropad>0
           dataloc=[dataloc;zeros(zeropad,nchan)];
        end
           
        datalocfft=fft(dataloc.*mywindow);
        
        
        
         for f=1:maxfreqbin % for all frequencies
          if npat==0
             if segave==0
                 cs(:,:,f,i)=cs(:,:,f,i)+conj(datalocfft(f,:)'*datalocfft(f,:)); 
		 av(:,f,i)=av(:,f,i)+conj(datalocfft(f,:)');
             else 
                %disp([i,f,size(datalocfft)])
                cs(:,:,f)=cs(:,:,f)+conj(datalocfft(f,:)'*datalocfft(f,:)); 
		av(:,f)=av(:,f)+conj(datalocfft(f,:)');
             end
          else 
             if segave==0
                 cs(:,:,f,j,i)=conj(datalocfft(f,:)'*datalocfft(f,:));
                 av(:,f,j,i)=conj(datalocfft(f,:)');  
             else 
                %disp([i,f,size(datalocfft)])
                cs(:,:,f,j)=cs(:,:,f,j)+conj(datalocfft(f,:)'*datalocfft(f,:));
                av(:,f,j)=av(:,f,j)+conj(datalocfft(f,:)');  
             end
          end

        end
    end
    nave=nave+1;
end

if segave==0
  cs=cs/nave;
  av=av/nave;
else
  nave=nave*nseg;  
  cs=cs/nave;
  av=av/nave;
end

for f=1:maxfreqbin
  if subave==1
       if npat==0
          if segave==0
              for i=1:nseg;cs(:,:,f,i)=cs(:,:,f,i)-av(:,f,i)*av(:,f,i)';end;
          else 
              cs(:,:,f)=cs(:,:,f)-av(:,f)*av(:,f)';
          end
       else 
          if segave==0
              for i=1:nseg;for j=1:nep;
                  cs(:,:,f,j,i)=cs(:,:,f,j,i)-av(:,f,j,i)*av(:,f,j,i)';
              end;end;
          else 
              for j=1:nep;cs(:,:,f,j)=cs(:,:,f,j)-av(:,f,j)*av(:,f,j)';end
          end
       end
  end
end

coh = [];

ndim=length(size(cs));
if ndim==3;
    [n1,n2,n3]=size(cs);
    coh=cs;
    for i=1:n3;
        c=squeeze(cs(:,:,i));
        coh(:,:,i)=c./sqrt(diag(c)*diag(c)');
    end
elseif ndim==4
    [n1,n2,n3,n4]=size(cs);
    coh=cs;
    for i=1:n3;for j=1:n4;
        c=squeeze(cs(:,:,i,j));
        coh(:,:,i,j)=c./sqrt(diag(c)*diag(c)');
    end;end;
end
    
   
    
    
    
    
    
    
return;


