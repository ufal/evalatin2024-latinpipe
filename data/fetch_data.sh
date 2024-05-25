#!/bin/bash
#
# This file is part of LatinPipe EvaLatin 24
# <https://github.com/ufal/evalatin2024-latinpipe>.
#
# Copyright 2024 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Refresh conllu_to_text.pl
curl -O https://raw.githubusercontent.com/UniversalDependencies/tools/master/conllu_to_text.pl

# Get the UD 2.13 data
curl -O https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz
tar xf ud-treebanks-v2.13.tgz

for tb in ud-treebanks-v2.13/*Latin*/*-ud-train.conllu; do
  dir=`dirname $tb`
  long_name=`basename $dir`
  code=`basename $tb`
  code=${code%%-*}
  echo Fetching $long_name as $code

  mkdir -p $code
  cp $dir/* $code
  if [ ! -f $code/$code-ud-dev.conllu ]; then
    python3 conllu_split.py $dir/$code-ud-train.conllu $code/$code-ud-train.conllu $code/$code-ud-dev.conllu
    for conllu in $code/$code-ud-train.conllu $code/$code-ud-dev.conllu; do
      perl conllu_to_text.pl --language=$code <$conllu >${conllu%.conllu}.txt
    done
  fi
done
rm -rf ud-treebanks-v2.13.tgz ud-treebanks-v2.13/

for dir in */; do
  echo "${dir%/}	$(grep -cP "^\d+\t" $dir*train.conllu)"
done | sort -rnk2 >langs_sizes

# Get the non-UD small datasets
corpus=la_sabellicus
echo Fetching $corpus
mkdir -p $corpus
wget https://raw.githubusercontent.com/CIRCSE/Sabellicus/main/Sabellicus_DeLatinaeLinguaeReparatione.conllu -O $corpus/$corpus-train.conllu

corpus=la_archimedes
echo Fetching $corpus
mkdir -p $corpus
wget https://raw.githubusercontent.com/mfantoli/ArchimedesLatinus/main/training_data_final_git.conllu -O $corpus/tmp1.conllu
wget https://raw.githubusercontent.com/mfantoli/ArchimedesLatinus/main/spirals_XIX_XX_test_git.conllu -O $corpus/tmp2.conllu
cat $corpus/tmp1.conllu $corpus/tmp2.conllu > $corpus/$corpus-train.conllu
rm $corpus/tmp1.conllu $corpus/tmp2.conllu

# Get the test data
wget https://github.com/CIRCSE/LT4HALA/raw/db51eaa114f437ac5c6cc04e802fc50cbd8b9f67/2024/data_and_doc/EvaLatin_2024_Syntactic_Parsing_test_data.zip
mkdir -p evalatin24
unzip -j EvaLatin_2024_Syntactic_Parsing_test_data.zip -d evalatin24/ -x "*/.*"
rm EvaLatin_2024_Syntactic_Parsing_test_data.zip

# Get harmonized PROIEL
mkdir -p la_proielh/
for split in train dev test; do
  wget https://github.com/fjambe/Latin-variability/raw/98f676c8a4575aee24c66fee73a09aecbe515643/morpho_harmonization/morpho-harmonized-treebanks/UD_Latin-PROIEL/MM-la_proiel-ud-$split.conllu -O la_proielh/la_proielh-ud-$split.conllu
done
