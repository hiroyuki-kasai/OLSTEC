# OLSTEC : OnLine Low-rank Subspace tracking by TEnsor CP Decomposition in Matlab

----------

Authors: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/)

Last page update: Sep. 12, 2017

Latest library version: 1.0.1 (see Release notes for more info)

Introduction
----------
**OLSTEC** is an online tensor subspace tracking algorithm based on the [Canonical Polyadic decomposition](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) (CP decomposition) 
(or PARAFAC or CANDECOMP decomposition) exploiting the [recursive least squares](https://en.wikipedia.org/wiki/Recursive_least_squares_filter) (RLS).

Motivation
----------
OLSTEC presents a new online tensor tracking algorithm for the partially observed high-dimensional data stream corrupted by noise. We focus on the fixed-rank higher-order [matrix completion](https://en.wikipedia.org/wiki/Matrix_completion) (i.e., tensor completion) algorithm with a second-order stochastic gradient descent based on the CP decomposition
exploiting the recursive least squares (RLS). Specifically, we consider the case where the partially observed tensor slice is acquired sequentially over time. Then, we estimate {A, B, C} by minimizing the exponentially weighted least squares defined as

<img src="http://www.kasailab.com/Public/Github/OLSTEC/images/problem_formulation.png" width="800">


Reference
---------
- H.Kasai, "Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations," Neurocomputing, Volume 347, 28, pp. 177-190, 2019.
    - [Publisher's web site](https://www.sciencedirect.com/science/article/abs/pii/S0925231218313584)
    - [arXiv web site](https://arxiv.org/abs/1709.10276)
- H.Kasai, "Online low-rank tensor subspace tracking from incomplete data by CP decomposition using recursive least squares," IEEE International conference on Acoustics, Speech and Signal Processing (ICASSP), 2016.
    - [Publisher's web site](http://ieeexplore.ieee.org/document/7472131/)
    - [arXiv web site](https://arxiv.org/abs/1602.07067)

List of benchmarks
---------
- Online tensor tracking algorithm
    - **TeCPSGD**
        - M. Mardani, G. Mateos, and G.B. Giannakis, "[Subspace learning and imputation for streaming big data matrices and tensors](http://ieeexplore.ieee.org/abstract/document/7072498/)," IEEE Transactions on Signal Processing, vol. 63, no. 10, pp. 266-2677, 2015.
- Online matrix tracking algorithms
    - [**Grasta**](https://sites.google.com/site/hejunzz/grasta)
        - Jun He, Laura Balzano, and John C.S. Lui, "[Online robust subspace tracking from partial information](https://arxiv.org/abs/1109.3827)," arXiv:1109.3827, 2011.
        - Jun He, Laura Balzano, and Arthur Szlam, "[Incremental gradient on the grassmannian for online foreground and background separation in subsampled video](http://ieeexplore.ieee.org/abstract/document/6247848/)," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
    - [**Grouse**](http://sunbeam.ece.wisc.edu/grouse/) 
        - L. Balzano, R. Nowak, and B. Recht, "[Online identification and tracking of subspaces from highly incomplete information](https://arxiv.org/abs/1006.4046)," arXiv:1006.4046, 2010.
    - [**Petrels**](http://www2.ece.ohio-state.edu/~chi/papers/petrels_codes.zip)
        - Y. Chi, Y. C. Eldar, and R. Calderbank, "[Petrels: Parallel subspace estimation and tracking using recursive least squares from partial observations](http://ieeexplore.ieee.org/document/6605610/)," IEEE Transactions on Signal Processing, vol. 61, no. 23, pp. 5947-5959, 2013.
- Batch tensor CP decomposition algorithm
    - **CP-WOPT**
        - E. Acar, D. M. Dunlavy, T. G. Kolda, and M. Mprup, "[Scalable tensor factorizations with missing data](http://www.sandia.gov/~dmdunla/publications/AcDuKoMo10.pdf)," Proceedings of the 2010 SIAM International Conference on Data Mining (SDM10), 2010, pp. 701-712.

Folders and files
---------

<pre>
./                              - Top directory.
./README.md                     - This readme file.
./olstec.m                      - OLSTEC algorithm file.
./run_me_first.m                - The scipt that you need to run first.
./demo.m                        - Demonstration script to check and understand this package easily. 
./test_comparison_syntheric.m   - Demonstration script for synthetic dataset. 
./test_comparison_real.m        - Demonstration script for real dataset. 
|auxiliary/                     - Some auxiliary tools for this project.
|benchmark/                     - Project files for benchmarks.
|tool/                          - 3rd party tools.
</pre>
- 3rd party tools
    - [tensor_toolbox_2.6](http://www.sandia.gov/~tgkolda/TensorToolbox/thankyou-2.6.html) and [poblano_toolbox_1.1](https://software.sandia.gov/trac/poblano) for CP-WOPT.
                                 

First to do
----------------------------
Run `run_me_first` for path configurations. 
```Matlab
%% First run the setup script
run_me_first; 
```

Usage example: Syhtethic dataset demo 
----------------------------
Now, just execute `demo` for demonstration of this package.
```Matlab
%% Execute the demonstration script
demo; 
```

The "**demo.m**" file contains below.
```Matlab
% set paramters
tensor_dims = [100, 100, 200];
rank        = 5;
fraction    = 0.1;
inverse_snr = 1e-4;

% generate tensor
data_subtype = 'Static';
[A,~,~,Omega,~,~,~,~,~,~,~,~] = generate_synthetic_tensor(tensor_dims, rank, fraction, inverse_snr, data_subtype);


% OLSTEC
options.verbose = 2;
[Xsol_olstec, infos_olstec, sub_infos_olstec] = olstec(A, Omega, [], tensor_dims, rank, [], options);


% plotting
figure;
semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_residual, '-r', 'linewidth', 2.0);
legend('OLSTEC');
xlabel('data stream index');
ylabel('normalized residual error');    


figure;
semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_run_ave, '-r', 'linewidth', 2.0);
legend('OLSTEC');
xlabel('data stream index');
ylabel('running average error');   
```

* Output results 

<img src="http://www.kasailab.com/public/github/OLSTEC/images/demo_result.png" width="900">
<br /><br />


More results
----------------------------

- Real-world dataset with moving background

    - The input video is created virtually by moving cropped partial image from its original entire frame image of video of "Airport Hall". 
    - The cropping window with 288x200 moves from the leftmost
partial image to the rightmost, then returns to the leftmost image after stopping a certain period of time. 
    - The generated video includes right-panning video from 38-th to 113-th frame and from 342-th to 417-th frame, and left-panning video from 190-th to 265-th frame. 
    - results
        - Normalized residual error and running average error.
        - Input image, caluculated low-rank image, and residual error image at 283-th frame.

<br />
<img src="http://www.kasailab.com/public/github/OLSTEC/images/real_datadata_dynamic_result_new.png" width="900">
<br />

<img src="http://www.kasailab.com/public/github/OLSTEC/images/dynamic_screen_shot.png" width="900">
<br />

License
-------
This code is free and open source for academic/research purposes (non-commercial).


Problems or questions
---------------------
If you have any problems or questions, please contact the author: [Hiroyuki Kasai](http://kasai.comm.waseda.ac.jp/kasai/) (email: hiroyuki **dot** kasai **at** waseda **dot** jp)

Release Notes
--------------
* Version 1.0.1 (Sep 12, 2017)
    - Bug fixed.
* Version 1.0.0 (June 07, 2017)
    - Initial version.

