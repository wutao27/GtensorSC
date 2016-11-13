# General Tensor Spectral Co-clustering for Higher-Order Data

#### Tao Wu
#### Austin R. Benson
#### David F. Gleich
------
### Folders
* `data` contains the info for generating tensor data. Please refer to the `README.md` file in the folder.
* `demo` contains IJulia notebook codes. Please download/process the data first.
* `script` contains the python script to generate the tensor data.

### Files
* `mymatrixfcn.jl` is the function for mat-vec (it is deployed by other functions).
* `shift_fixpoint.jl` are the functions computing the stationary distribution for the super-spacey random surfer.
* `tensor_cut` are the functions computing the biased conductance and sweep cut.
* `util.jl` contains the main function for recursive two-way cuts and some utility functions.

### Usage
Please refer to the demos.
Note: The codes should be compatible with Julia 0.4 and 0.3. If you get errors when using Julia 0.5, please try 0.4 or 0.3.
