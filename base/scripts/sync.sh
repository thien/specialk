# to be called from the azure instance.
cd ..
rsync -av models/ t@casper.thien.io:/media/data/azure_dumps
python3 core/telegram.py -m "Finished transferring files to local machine. You need to turn off the instance!"