from __future__ import division
from numpy import *
from matplotlib.mlab import *
from circ_stats import *
from scipy.stats import *
from scikits.bootstrap import ci
from scikits import bootstrap
#import statsmodels.nonparametric.smoothers_lowess as loess
from constants import *
#import seaborn as sns
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
import pandas as pd
import sys
from scipy import special
find = lambda x: where(x)[0]
import seaborn as sns
#import heike_helpers as hf


#set_printoptions(precision=4)
#sns.set_context("talk", font_scale=1.3)
#sns.set_style("ticks")
#sns.set_style({"ytick.direction": "in"})


def boot_test(data,thr=0,n_samples=1000000):
	data=array(data)
	t_data = nanmean(data) - thr
	boot_data = data[array(bootstrap.bootstrap_indexes(data,n_samples=n_samples))]
	t_boot = (nanmean(boot_data,1) - nanmean(data)) 
	p =  nanmean(abs(t_data)<=abs(t_boot))
	return p,percentile(nanmean(boot_data,1),[2.5,97.5])

def boot_test1(data,thr=0,n_samples=1000000):
	data=array(data)
	t_data = nanmean(data) - thr
	boot_data = data[bootstrap.bootstrap_indexes_array(data,n_samples=n_samples)]
	t_boot = (nanmean(boot_data,1) - nanmean(data)) 
	low =  nanmean(t_data<=t_boot)
	high =  nanmean(t_data>=t_boot)
	return low,high,percentile(nanmean(boot_data,1),[2.5,97.5])

def perm_test(data_a,data_b,n_perms=10000):
    r_d = mean(data_a - data_b,0)
    data_a = data_a.copy()
    data_b=data_b.copy()
    D=[]
    for _ in range(n_perms):
        idx=where(rand(len(data_a))<0.5)[0]
        data_a[idx],data_b[idx]=data_b[idx],data_a[idx]
        d=mean(data_a - data_b,0)
        D.append(d)
    return mean(r_d < array(D),0)*2

def perm_test_nan(data_a,data_b,n_perms=10000):
    r_d = nanmean(data_a,0) - nanmean(data_b,0)
    data_a = data_a.copy()
    data_b=data_b.copy()
    D=[]
    for _ in range(n_perms):
        idx=where(rand(len(data_a))<0.5)[0]
        data_a[idx],data_b[idx]=data_b[idx],data_a[idx]
        d=nanmean(data_a,0) - nanmean(data_b,0)
        D.append(d)
    return mean(r_d < array(D),0)*2

def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	for s in sigs:
		beg =axis[s]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

def perm_test2(data_a,data_b,n_perms=10000):
	n_a = len(data_a)
	r_d = nanmean(data_a) - nanmean(data_b)
	data_a = data_a.copy()
	data_b=data_b.copy()
	both = concatenate([data_a,data_b])
	D=[]
	for _ in range(n_perms):
		#idx=where(rand(len(both))<0.5)[0]
		shuffle(both)
		data_a,data_b=both[:n_a],both[n_a:]
		d=nanmean(data_a) - nanmean(data_b)
		D.append(d)
	return nanmean(r_d < array(D),0)*2



def circ_mean(x):
	return circmean(x,low=-pi,high=pi)

def compute_seria_from_pandas(pandas,xxx,flip=None):
	return compute_serial(pandas.response.values,pandas.target.values,
		pandas.prevcurr,xxx,flip)

def compute_serial(report,target,d,xxx,flip=None):
	n=0
	err=circdist(report,target)
	m_err=[]
	std_err=[]
	count=[]
	cis=[]
	uf_err = err.copy()
	if flip:
		err = sign(d)*err
		d=abs(d)
	points_idx=[]
	for i,t in enumerate(xxx):
		# wi=w[i]
		idx=(d>=t)&(d<=t+w2)
		m_err.append(circ_mean(err[idx]))
		std_err.append(circstd(err[idx])/sqrt(sum(idx)))
		count.append(sum(idx))
		points_idx.append(idx)
	return [array(err),d,array(m_err),array(std_err),count,points_idx,n,uf_err]

def plot_sigs(all_s_a,color,all_s_b=[],upper=[3.75,4],alpha=0.05,pvalues=[],xx=None):
	if xx is None:
		xx = xxx2
	if len(pvalues)==0:
		n_perms = 10000
		n_subjects_a = len(all_s_a)
		if len(all_s_b):
			ci_sb = array([ci(sb,method="pi",n_samples=10000,alpha=alpha) for sb in (all_s_a-all_s_b).T])
		else:
			ci_sb = array([ci(sb,method="pi",n_samples=10000,alpha=alpha) for sb in (all_s_a).T])
		sigs = find((ci_sb[:,0]>0) | ((ci_sb[:,1])<0))
	else:
		sigs = find(array(pvalues)<alpha)
	ylim(-2,upper[1])
	sig_bar(sigs,xx,upper,color)

def plot_serial(all_s,color,label=None,xx=None):
	mean = nanmean
	if xx is None:
		xx = xxx2
	stderr = array([ci(sb,statfunction=mean,alpha=1-0.68,method="pi") for sb in (all_s).T])
	fill_between(xx,degrees(stderr[:,0]),degrees(stderr[:,1]),color=color,alpha=0.2,label=label)

	plot(xx,degrees(mean(all_s,0)),color=color)
	plot(xx,zeros(len(xx)),"k--",alpha=0.5)

	#xlabel(r"relative location ($^\circ$)")
	ylabel(r"current error ($^\circ$)")
	#legend()
	sns.despine()


#######################################################################

def vonmisespdf(x, mu, k):
	# modified bessel function of first kind and order 0
	return exp(k*cos(x-mu)) / (2*pi * special.iv(0,k))

#######################################################################
