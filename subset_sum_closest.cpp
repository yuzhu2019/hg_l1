#include <math.h>
#include <stdio.h>
#include <string>
#include <random>
#include <iostream>
#include "mex.h"

unsigned int count_nonzeros(unsigned int * arr, unsigned int n){
    unsigned int count = 0;
    for (unsigned int i=0; i<n; i++) {
        if (arr[i] != 0)
            count++;
    }
    return count;
}

void find_nonzeros(unsigned int * output, unsigned int * input, unsigned int n){
    unsigned int j = 0;
    for (unsigned int i=0; i<n; i++) {
        if (input[i] != 0) {
            output[j] = i;
            j++;
        }
    }
}

std::vector<unsigned int> subset_sum_closest_dp_int(unsigned int * nums, unsigned int target, unsigned int n){
    
    unsigned int ** records = new unsigned int * [n];
    unsigned int * possible_sums = new unsigned int[target+1]();
    unsigned int i, j, len, curr;
    
    for(i=0; i<n; i++) {
        len = count_nonzeros(possible_sums, target+1);
        unsigned int * possible_sums_keys = new unsigned int[len]();
        find_nonzeros(possible_sums_keys, possible_sums, target+1);
        for(j=0; j<len; j++) {
            curr = nums[i] + possible_sums_keys[j];
            if(curr <= target)
                possible_sums[curr] = i + 1;
        }
        if(nums[i] <= target)
            possible_sums[nums[i]] = i + 1;
        records[i] = new unsigned int[target+1]();
        for(j=0; j<=target; j++)
            records[i][j] = possible_sums[j];
        delete[] possible_sums_keys;
    }
    
    len = count_nonzeros(possible_sums, target+1);
    unsigned int * possible_sums_keys = new unsigned int[len]();
    find_nonzeros(possible_sums_keys, possible_sums, target+1);

    unsigned int best_sum = possible_sums_keys[0];
    for (i=1; i<len; i++) {
        if(possible_sums_keys[i] > best_sum)
            best_sum = possible_sums_keys[i];
    }
    
    unsigned int record_idx = n - 1;
    unsigned int last_idx;
    std::vector<unsigned int> res;
    while(best_sum) {
        last_idx = records[record_idx][best_sum] - 1;
        res.push_back(last_idx);
        best_sum = best_sum - nums[last_idx];
        record_idx = last_idx - 1;
    }
    
    delete[] possible_sums;
    delete[] possible_sums_keys;
    
    for(i=0; i<n; i++)
        delete[] records[i];
    delete[] records;
    
    return res;
}

double subset_sum_closest_dp(double * nums, double target, unsigned int n, double scale){
    unsigned int i;
    unsigned int * nums_proj = new unsigned int[n];
    for(i=0; i<n; i++)
        nums_proj[i] = (unsigned int) ceil(scale * nums[i]);
    unsigned int target_proj = (unsigned int) ceil(scale * target);
    std::vector<unsigned int> pos;
    pos = subset_sum_closest_dp_int(nums_proj, target_proj, n);
    unsigned int pos_len = pos.size();
    double res = 0;
    for (i=0; i<pos_len; i++)
        res = res + nums[pos[i]];
    delete[] nums_proj;
    return res;
}

void find(std::vector<double>&v, int i, int e, double sum, std::vector<double>&sumv){
    if(i == e) {
        sumv.push_back(sum);
        return;
    }
    find(v, i+1, e, sum+v[i], sumv);
    find(v, i+1, e, sum, sumv);
}
    
double subset_sum_closest_bf(double * nums, double target, int n){
        
    std::vector<double>A,B;
    for(int i=0; i<n/2; i++)
        A.push_back(nums[i]);
    for(int i=n/2; i<n; i++)
        B.push_back(nums[i]);
    
    std::vector<double>sumA,sumB;
    find(A, 0, A.size(), 0, sumA);
    find(B, 0, B.size(), 0, sumB);

    std::sort(sumA.begin(), sumA.end());
    std::sort(sumB.begin(), sumB.end());
    
    double ans = INT_MAX;
    double res;
    for(int i=0; i<sumA.size(); i++) {
        double s = sumA[i];
        int l = 0;
        int r = sumB.size() - 1;
        while(l <= r) {
            int mid = l+(r-l)/2;
            double sum = s + sumB[mid];
            if(sum == target)
                return sum;
            if(sum > target) {
                if(sum - target < ans) {
                    ans = sum - target;
                    res = sum;
                }
                r = mid - 1;
            }
            else {
                if(target - sum < ans) {
                    ans = target - sum;
                    res = sum;
                }
                l = mid + 1;
            }
        }
    }
    return res;
}

void main_bf(double * output, double * nums, double target, int n){
    double res = subset_sum_closest_bf(nums, target, n);
    *output = res;
}

void main_dp(double * output, double * nums, double target, unsigned int n, double scale){
    double res = subset_sum_closest_dp(nums, target, n, scale);
    *output = res;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if (nlhs != 1 || nrhs != 3) {
        mexWarnMsgTxt("Check Parameters");
        return;
    }
    // input
    double * nums = mxGetPr(prhs[0]);
    double target = mxGetScalar(prhs[1]);
    unsigned int n = (unsigned int) mxGetScalar(prhs[2]);
    // output
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double * output = mxGetPr(plhs[0]);
    // alg
    if(n <= 45)
        main_bf(output, nums, target, (int) n);
    else if(n <= 500)
        main_dp(output, nums, target, n, 10000);
    else
        main_dp(output, nums, target, n, 1000);
}


