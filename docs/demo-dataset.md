# Demonstration Dataset

**NOTE 2018-10-18:** we are in the process of improving the heuristic agent (bot) and will be releasing a new dataset of higher-quality demonstrations soon.

Generating demonstrations takes a sizeable amount of computational resources. A gzipped archive containing the demonstrations used for the ICLR 2019 submission is [available here](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/chevalma/iclr19-demos.tar.gz) (14GB download). Please note that these demonstrations can only be used with the ICLR 2019 [docker image](https://github.com/mila-iqia/babyai#docker-image) as they are no longer compatible with the source code on the master branch of this repository. If you wish to work with latest BabyAI source code, you should generate a new demonstration dataset.

Once downloaded, extract the `.pkl` files to `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>/demos`.