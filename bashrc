# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias sshm01='ssh -p 33345 djzhou@117.187.66.180'
# Beijing ssh djzhou@10.14.2.84
alias sshmgt01='ssh djzhou@10.14.2.84'
alias laptopzdj='ssh -X zdj@2400:dd01:102d:1001:44f6:b6a3:be8e:d01a'
alias asuszdj='ssh -X zdj@2400:dd01:103a:2020:d98e:8995:8099:11ef'
alias jwc='ssh -X jingweicong@10.14.2.28'
alias cnn='ssh -X cnn@2400:dd01:102d:14:5518:21ad:e93a:c732'
#PSRSOFT
export PSRSOFT_USR=/home/zhoudejiang/psrsoft/usr
for env in $PSRSOFT_USR/var/psrsoft/env/bash/* ; do . $env ; done
export PATH=$PATH:$PSRSOFT_USR/bin
export LD_LIBRARY_PATH=$PSRSOFT_USR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PSRSOFT_USR/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$PSRSOFT_USR/include:$C_INCLUDE_PATH
# export TEMPO=$PSRSOFT_USR/src/tempo_1
# export PATH=/home/zhoudejiang/psrsoft/usr/src/tempo_1/build/bin/:$PATH

#presto
export ASTROSOFT=/home/zhoudejiang/psrsoft/usr
export PRESTO=/home/zhoudejiang/work/FAST_DATA_processing/presto_20220214
export PATH=$PRESTO/bin:$PATH
export LD_LIBRARY_PATH=$PRESTO/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PRESTO/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$PRESTO/include:$C_INCLUDE_PATH
export PYTHONPATH=/home/zhoudejiang/anaconda3/lib/python3.8/site-packages:/home/zhoudejiang/psrsoft/usr/src/presto/build/lib.linux-x86_64-3.8:/home/zhoudejiang/psrsoft/usr/src/presto/python:/home/zhoudejiang/psrsoft/usr/src/presto/python/presto:/home/zhoudejiang/psrsoft/usr/src/presto/python/presto_src:/home/pulsar/pulsar_software/lib/python2.7/site-packages


#libpng
export PATH=/home/zhoudejiang/soft/libpng-1.2.59/build/bin:$PATH
export C_INCLUDE_PATH=/home/zhoudejiang/soft/libpng-1.2.59/build/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/zhoudejiang/soft/libpng-1.2.59/build/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/zhoudejiang/soft/libpng-1.2.59/build/lib:$LIBRARY_PATH

#gsl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/include


#parallel
export PATH=/home/zhoudejiang/soft/parallel-20201022/build/bin:$PATH

#fv
export PATH=$PATH:/home/zhoudejiang/soft/fv5.5.2/

#cmake
export PATH=/home/zhoudejiang/Downloads/cmake-3.18.2-Linux-x86_64/bin:$PATH

#gcc
export PATH=/usr/bin:$PATH
#export PATH=/usr/local/gcc/bin:$PATH
#export C_INCLUDE_PATH=/usr/local/gcc/include:$C_INCLUDE_PATH
#export LD_LIBRARY_PATH=/usr/local/gcc/lib64:/usr/local/gcc/lib:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/usr/local/gcc/lib64:/usr/local/gcc/lib:$LIBRARY_PATH

#glib2
export C_INCLUDE_PATH=/usr/include/glib-2.0/:/usr/lib64/glib-2.0/include/:$C_INCLUDE_PATH


#CUDA
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/usr/local/cuda-11.1/include:$C_INCLUDE_PATH

#PGPLOT
export PGPLOT_DIR=/usr/local/pgplot/
export PGPLOT_DEV=/xwin	  # prefered output device, an alternative is /xserve
export PGPLOT_FONT=/usr/local/pgplot/grfont.dat
export PGPLOT_RGB=/usr/local/pgplot/rgb.txt

#mysoft
export PATH=$PATH:/home/zhoudejiang/Myselfbash/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/lib64
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib64:/usr/lib64

#darknet
export DARKNET=/home/zhoudejiang/soft/darknet/

#pkg
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib64/pkgconfig:/usr/local/lib64/pkgconfig

#latex
PATH=/usr/local/texlive/2020/bin/x86_64-linux:$PATH
MANPATH=/usr/local/texlive/2020/texmf-dist/doc/man:$MANPATH
INFOPATH=/usr/local/texlive/2020/texmf-dist/doc/info:$INFOPATH



export PATH=/home/zhoudejiang/anaconda3/bin:$PATH
export LD_RUN_PATH=$LD_RUN_PATH:$LD_LIBRARY_PATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!

#__conda_setup="$('/home/zhoudejiang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/zhoudejiang/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/zhoudejiang/anaconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/zhoudejiang/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup

#__conda_setup="$('/home/zhoudejiang/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/zhoudejiang/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/zhoudejiang/anaconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/zhoudejiang/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup

# <<< conda initialize <<<


#psrsalsa
export PATH=$PATH:/home/zhoudejiang/soft/psr-psrsalsa-build/label/stretch/psrsalsa/bin

#code
# export PATH=$PRESTO/bin:$PATH:/home/zhoudejiang/soft/code/VSCode-linux-x64/bin

#mpicc
MPI_ROOT=/usr/lib64/mpich
export PATH=$MPI_ROOT/bin:$PATH

# ffancy
export PATH=$PATH:/home/zhoudejiang/work/FAST_DATA_processing/ffancy/bin


#boost
export LD_LIBRARY_PATH=/home/zhoudejiang/work/FAST_DATA_processing/iqrm_pkg/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/zhoudejiang/work/FAST_DATA_processing/iqrm_pkg/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/home/zhoudejiang/software/boost/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/zhoudejiang/software/boost/include:$CPLUS_INCLUDE_PATH

#cmake
export PATH=/home/zhoudejiang/soft/cmake-3.10.2-Linux-x86_64/bin/:$PATH

#iqrm
export PATH=/home/zhoudejiang/work/FAST_DATA_processing/iqrm_pkg/bin:$PATH

#pperf
# export PPROF_PATH=/usr/bin/pprof


#code
export PATH=/home/zhoudejiang/Downloads/VSCode-linux-x64/bin:$PATH

