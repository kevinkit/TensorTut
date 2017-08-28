Some reasons why Tensorflow could not work (with GPU) under Windows might be:


1. Microsoft Visual Studio was not installed, or the installed version is not the supported one from TF
   - Currently it has to be Visual Studio 2015, with Ugprade 3

2. The cuDNN version is not suitable
   - The current version should be 6

3. The CUDA Version does not fit
   - CUDA 8.0 should be installed

4. you may need to pip install tensorflow and then pip install tensorflow-gpu to get the GPU version to run

5. Spyder should be installed for all users, and pip comamnds should be envoked from an admin command line