#!/bin/bash

set -e

# OVERVIEW
# This script enables Jupyter to export a notebook directly to PDF.
# nbconvert depends on XeLaTeX and several LaTeX packages that are non-trivial to
# install because `tlmgr` is not included with the texlive packages provided by yum.

# REQUIREMENTS
# Internet access is required in order to fetch the below latex libraries from the ctan mirror.

sudo -u ec2-user -i <<EOF
unset SUDO_UID

mkdir -p /home/ec2-user/SageMaker/.texmf
cd /home/ec2-user/SageMaker/.texmf
wget http://mirrors.ctan.org/install/macros/latex/contrib/tcolorbox.tds.zip
wget http://mirrors.ctan.org/install/macros/latex/contrib/environ.tds.zip
wget http://mirrors.ctan.org/install/macros/latex/contrib/etoolbox.tds.zip
wget http://mirrors.ctan.org/install/macros/latex/contrib/trimspaces.tds.zip
wget http://mirrors.ctan.org/macros/latex/contrib/upquote.zip
unzip tcolorbox.tds.zip
unzip environ.tds.zip
unzip etoolbox.tds.zip
unzip trimspaces.tds.zip
unzip upquote.zip
mv upquote tex/latex/

EOF