- Clone nvbandwidth into this directory.
``` sh
git clone https://github.com/NVIDIA/nvbandwidth.git
```
- Patch nvbandwidth's CMake list to remove the use of static library.
- Build the image.
``` sh
docker build .
```