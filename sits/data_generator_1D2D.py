import numpy as np
import matplotlib.pyplot as plt
import mysrc.constants as cst

# -----------------------------------------------------------------------
def computingMinMaxPerSamplePerBand(Xt, D):
    Xt_shape = Xt.shape
    if D == 1:
        min_per_sample = np.reshape(np.amin(Xt, axis=1), (Xt_shape[0], 1, Xt_shape[2]))
        max_per_sample = np.reshape(np.amax(Xt, axis=1), (Xt_shape[0], 1, Xt_shape[2]))
    elif D == 2:
        min_per_sample = np.reshape(np.amin(Xt, axis=(1, 2)),(Xt_shape[0],1,1,Xt_shape[3]))
        max_per_sample = np.reshape(np.amax(Xt, axis=(1, 2)),(Xt_shape[0],1,1,Xt_shape[3]))
    return min_per_sample, max_per_sample
# -----------------------------------------------------------------------
def computingMinMaxPerBand(Xt, D):
    Xt_shape = Xt.shape
    if D == 1:
        min_per_sample = np.reshape(np.amin(Xt, axis=(0, 1)), (1, Xt_shape[2]))
        max_per_sample = np.reshape(np.amax(Xt, axis=(0, 1)), (1, Xt_shape[2]))
    elif D == 2:
        min_per_sample = np.reshape(np.amin(Xt, axis=(0, 1, 2)),(1,1,Xt_shape[3]))
        max_per_sample = np.reshape(np.amax(Xt, axis=(0, 1, 2)),(1,1,Xt_shape[3]))
    return min_per_sample, max_per_sample
# -----------------------------------------------------------------------
def normMinMax(X, min_per, max_per, back=False):
    if back == True:
        return X * (max_per - min_per) + min_per
    else:
        return (X - min_per) / (max_per - min_per)
# -----------------------------------------------------------------------
class DG(object):
    """
    Data augmentation. First data are systematically shifted left and right of xshift ( xshift = [1,2]). We thus have the orginal histo plus
    the 4 shifted (5 in total). On this we add gaussian noise (norm, gauss add, mask, norm back). In total we have 10 samples from 1 histo
    After that we add gaussian to the corresponding yield as well. This will double the samples (20 from 1)

    Assumption: histograms are raw data or mormalized betwen 0 and 0, i.e. zeros are true zeros (to be masked)
    """

    def __init__(self, Xt_full, region_ohe, y, Xshift=False, Xnoise=False, X_noiseSD=0.05,
                 Ynoise=False, Y_noiseSD=0.05):
        """Instantiates the class with metadata"""
        # Determine if it is 1D or 2 D
        if Xt_full.ndim == 4:
            self.D = 2
        elif Xt_full.ndim == 3:
            self.D = 1
        self.X = Xt_full         # the full set histograms to be augmented (must be complete histo, i.e. 36 dekads, shape (510 samples,64 y-bins,36 x-time-steps,4-bands)
        self.X_augmented = None
        self.region_ohe = region_ohe
        self.region_ohe_augmented = None
        self.y = y              # the var to be estimated, can be yield or area
        self.y_augmented = None
        self.Xshift = Xshift
        self.Xnoise = Xnoise
        self.X_noiseSD = X_noiseSD
        self.Ynoise = Ynoise
        self.Y_noiseSD = Y_noiseSD

    def generate(self, lenTS, subset_bool):
        # lenTS: the length of the time series to return (starting from index 3 that is 1st of Sep)
        # subset_bool: train samples to be augmented

        # set augmented arrays to the original data arrays
        self.X_augmented = self.X[subset_bool] #, :, :, :] works for 1d and 2d
        X_current = self.X_augmented.copy()
        self.region_ohe_augmented = self.region_ohe[subset_bool,:]
        #self.groups_augmented = self.groups[subset_bool]
        self.y_augmented = self.y[subset_bool]

        if self.Xshift == True:
            # 1 - shift left (no matter if we leave the last deks unchanged, they will not be used)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, -1, axis=self.D)), axis=0)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, -2, axis=self.D)), axis=0)
            # 2 - shift right
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, 1, axis=self.D)), axis=0)
            self.X_augmented = np.concatenate((self.X_augmented, np.roll(X_current, 2, axis=self.D)), axis=0)
            # add unchanged data for the other variables
            self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (5,1)) #repeat 5 times (original + 4 augmented) on axis 0
            self.y_augmented = np.tile(self.y_augmented, 5)
            if self.D == 1 and False:
                id2plt = 40  # id to plot (on non - augmented  self.X[subset_bool] which is equal to X_current
                deltaAug = X_current.shape[0]
                nAug = int(self.X_augmented.shape[0]/deltaAug-1)
                vars = ['NDVI', 'rad', 'rainfall', 'temperature']
                fig, axs = plt.subplots(1, 4, figsize=(16.5, 5))
                clr = ['Green', 'Purple', 'Blue', 'Red']
                for col in range(len(clr)):
                    ax = axs[col]
                    plt.sca(ax)
                    y_original = self.X_augmented[id2plt,:,col]
                    plt.plot(y_original, color=clr[col])
                    for na in range(nAug):
                        #plot augmented versions, exlude fir and last 2 dekads, that rae not used and effect of roll
                        y_aug = self.X_augmented[id2plt+deltaAug*(na+1), :, col]
                        plt.plot(y_aug, '--', color=clr[col], lw=0.5)
                        plt.xlim(2,33)
                    ax.set_title(vars[col])
                plt.show()
                print('debug')
                plt.close()

            if False:   #this is for checking 2d
                variables = ['NDVI', 'Radiation', 'Rainfall', 'Temperature']
                fig, axs = plt.subplots(2, 4, figsize=(16.5, 7))
                cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
                for col in range(len(cmaps)):
                    ax = axs[0,col]
                    plt.sca(ax)
                    pcm = ax.imshow(np.flipud( self.X_augmented[0,:, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    plt.title(variables[col])
                    ax = axs[1, col]
                    plt.sca(ax)
                    pcm = ax.imshow(np.flipud(self.X_augmented[510, :, :, col]), cmap=cmaps[col])
                    fig.colorbar(pcm, ax=ax)
                    plt.title(variables[col])
                plt.tight_layout()
                plt.show()

        if self.Xnoise == True:
            sz_before_da = self.X_augmented.shape[0]
            if self.D == 1:
                if True:
                    # We act differently on different variables
                    X0 = self.X_augmented.copy()
                    # NDVI: multiplicative error 10 % as in Copernicus global land target for fapar, 10
                    k = 0.05
                    # add noise (unit mean and 0.05 SD so that 99% of values are in +/- 0.1
                    standardNoise_1SD = np.random.normal(1, k, self.X_augmented[:, :, 0].shape)
                    tmp = standardNoise_1SD * self.X_augmented[:, :, 0]
                    # be care it can give negative values or values greater than 1 (potentially), clip to zeros if there were negative values
                    tmp[tmp < 0] = 0
                    tmp[tmp > 1] = 1
                    X0[:, :, 0] = tmp

                    # Radiation, additive noise 15% as extrapolated from https://www.nature.com/articles/s41598-020-60460-1
                    # RMSE di circa 30 Wm2 su una radiazione media che è di circa 200-250 Wm2 on momthly avarages (Fig 5), 15%
                    # as we have 10 days values the variance is about (https://en.wikipedia.org/wiki/Propagation_of_uncertainty)
                    #sd_sum_dek = sqrt(3) * sd_month.
                    #$ However this is RMSE, so with respect to observations, so there is a bias
                    # We have about 250,000 KJ/m2/dek, we take 10% would be 25000
                    Noise = np.random.normal(0, 20000, self.X_augmented[:, :, 1].shape)
                    tmp = self.X_augmented[:, :, 1] + Noise
                    tmp[tmp < 0] = 0
                    X0[:, :, 1] = tmp

                    # Rainfall, here multiplicative noise according to doi:10.1002/grl.50320, 2013
                    # here we assume a 10 % error
                    k = 0.1
                    # add noise (unit mean and 0.1SD so that 99% of values are in +/- 0.1
                    standardNoise_1SD = np.random.normal(1, k, self.X_augmented[:, :, 2].shape)
                    tmp = standardNoise_1SD * self.X_augmented[:, :, 2]
                    # be care it can give negative values, clip to zeros if there were negative values
                    tmp[tmp < 0] = 0
                    X0[:, :, 2] = tmp

                    # Temperature, additive noise according to  https://doi.org/10.5194/acp-18-8113-2018
                    # it is a bit of extrapolation because what they report is multi-annual error compared to observations
                    # we add a noise with SD - 1.5 deg
                    Noise = np.random.normal(0, 1.5, self.X_augmented[:, :, 3].shape)
                    tmp = self.X_augmented[:, :, 3] + Noise
                    X0[:, :, 3] = tmp
                    # Add to sample
                    self.X_augmented = np.concatenate((self.X_augmented, X0), axis=0)
                if False:
                    # we normalize 0-1, add noise, clip to >= 0 and back to value
                    # Normalize
                    min_per_band, max_per_band = computingMinMaxPerBand(self.X_augmented, self.D)
                    X0 = normMinMax(self.X_augmented, min_per_band, max_per_band)
                    # add noise
                    X0 = X0 + np.random.normal(0, self.X_noiseSD, X0.shape)
                    # adding noise can result in negative values, clip to zeros if there are negative values
                    X0[X0 < 0] = 0
                    # now denormalize back and add to augmented sample
                    self.X_augmented = np.concatenate(
                        (self.X_augmented, normMinMax(X0, min_per_band, max_per_band, back=True)), axis=0)
                # add data for the other variables
                # self.region_ohe_augmented = np.tile(self.region_ohe_augmented, 2)
                self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (2, 1))
                # self.groups_augmented = np.tile(self.groups_augmented, 2)
                self.y_augmented = np.tile(self.y_augmented, 2)
                if self.D == 1 and False:
                    id2plt = 2  # id to plot (on non - augmented  self.X[subset_bool] which is equal to X_current
                    vars = ['NDVI', 'rad', 'rainfall', 'temperature']
                    fig, axs = plt.subplots(1, 4, figsize=(16.5, 5))
                    clr = ['Green', 'Purple', 'Blue', 'Red']
                    for col in range(len(clr)):
                        ax = axs[col]
                        plt.sca(ax)
                        y_original = self.X_augmented[id2plt, :, col]
                        plt.plot(y_original, color=clr[col])
                        # plot augmented versions, exlude fir and last 2 dekads, that rae not used and effect of roll
                        y_aug = self.X_augmented[id2plt + sz_before_da, :, col]
                        plt.plot(y_aug, '--', color=clr[col], lw=0.5)
                        plt.xlim(2, 33)
                        ax.set_title(vars[col])
                    plt.show()
                    print('debug')
            elif self.D == 2:
                # X data can come normalized min max (min hard coded to 0) to 0-1 (so min is actually 0 count) or not
                # so I normalize again here (if it is already norm has no effect).
                # I have to normalize count (0 to n)  in [0,1] to apply a gaussian noise with 0 mean and SD
                # But we don't want to add noise in 0 count grid cell, so I have to mask the zeros and keep them zeros
                # Normalize
                min_per_sample, max_per_image = computingMinMaxPerSamplePerBand(self.X_augmented, self.D)
                X0 = normMinMax(self.X_augmented, min_per_sample, max_per_sample)
                # add noise
                X0 = X0 + np.random.normal(0, self.X_noiseSD, X0.shape)
                # set back to zero those that were 0
                X0[self.X_augmented == 0] = 0
                # adding noise can result in negative values, clip to zeros if there are negative values
                X0[X0 < 0] = 0
                # now denormalize back and add to augmented sample
                self.X_augmented = np.concatenate((self.X_augmented, normMinMax(X0, min_per_sample, max_per_sample, back=True)), axis=0)
                # add data for the other variables
                #self.region_ohe_augmented = np.tile(self.region_ohe_augmented, 2)
                self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (2, 1))
                #self.groups_augmented = np.tile(self.groups_augmented, 2)
                self.y_augmented = np.tile(self.y_augmented, 2)
                if False:
                    id2plt = 1000 # refers to a sample before adding noise (must be < n_before_noise)
                    fig, axs = plt.subplots(3, 4, figsize=(16.5, 8))
                    cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
                    for col in range(len(cmaps)):
                        ax = axs[0,col]
                        plt.sca(ax)
                        pcm = ax.imshow(np.flipud(self.X_augmented[id2plt,:, :, col]), cmap=cmaps[col])
                        fig.colorbar(pcm, ax=ax)
                        zeros = self.X_augmented[id2plt, :, :, col].copy()
                        zeros[zeros !=0] = np.nan
                        zeros[zeros == 0] = 1
                        pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                        variables = ['NDVI', 'Radiation', 'Rainfall', 'Temperature']
                        plt.title(variables[col])

                        ax = axs[1, col]
                        #pcm = ax.imshow(np.flipud(X_noisy[id2plt, :, :, col]), cmap=cmaps[col])plt.sca(ax)
                        pcm = ax.imshow(np.flipud(self.X_augmented[id2plt+n_before_noise, :, :, col]), cmap=cmaps[col])
                        fig.colorbar(pcm, ax=ax)
                        zeros = self.X_augmented[id2plt+n_before_noise, :, :, col].copy() #X_noisy[id2plt, :, :, col].copy()
                        zeros[zeros != 0] = np.nan
                        zeros[zeros == 0] = 1
                        pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                        variables = ['noisy_NDVI', 'noisy_Radiation', 'noisy_Rainfall', 'noisy_Temperature']
                        plt.title(variables[col])

                        ax = axs[2, col]
                        plt.sca(ax)
                        #noise = X_noisy[id2plt,:, :, col] - self.X_augmented[id2plt,:, :, col]
                        noise = self.X_augmented[id2plt+n_before_noise, :, :, col] - self.X_augmented[id2plt, :, :, col]
                        pcm = ax.imshow(np.flipud(noise), cmap=cmaps[col])
                        fig.colorbar(pcm, ax=ax)
                        zeros = noise.copy()
                        zeros[zeros != 0] = np.nan
                        zeros[zeros == 0] = 1
                        pcm = ax.imshow(np.flipud(zeros), cmap='gray')
                        variables = ['noise_NDVI', 'noise_Radiation', 'noise_Rainfall', 'noise_Temperature']
                        plt.title(variables[col])
                    plt.tight_layout()
                    plt.show()
        if self.Ynoise == True:
            sz_before_da = self.y_augmented.shape[0]
            # here, in the absense of information about it, we assume that the error is proportional to yield value (multiplicative noise)
            if True:
                # add noise proportional to signal
                # self.Y_noiseSD is the proportionality factor, 0.1 means 10% of signal
                # add noise
                standardNoise_1SD = np.random.normal(0, 1, self.y_augmented.shape)
                multiplicativeNoise = standardNoise_1SD * self.Y_noiseSD * self.y_augmented
                y0 = self.y_augmented + multiplicativeNoise
                # be care it can give negative values, clip to zeros if there were negative values
                y0[y0 < 0] = 0
                # Add to sample
                self.y_augmented = np.concatenate((self.y_augmented, y0), axis=0)
            if False:
                # add addittive noise
                # as we did for X we normalize 0-1, add noise, clip to >= 0 and back to value
                # Normalize
                min_y = np.amin(self.y_augmented, axis=0)
                max_y = np.amax(self.y_augmented, axis=0)
                y0 = normMinMax(self.y_augmented, min_y, max_y)
                # add noise
                y0 = y0 + np.random.normal(0, self.Y_noiseSD, y0.shape)
                # be care it can give negative values, clip to zeros if there were negative values
                y0[y0 < 0] = 0
                # now denormalize back and add to sample
                self.y_augmented = np.concatenate((self.y_augmented, normMinMax(y0, min_y, max_y, back=True)), axis=0)


            # add data for the other variables
            self.region_ohe_augmented = np.tile(self.region_ohe_augmented, (2, 1))
            if self.D == 1:
                self.X_augmented = np.tile(self.X_augmented, (2, 1, 1))
            elif self.D == 2:
                self.X_augmented = np.tile(self.X_augmented, (2, 1, 1, 1))
            if self.D == 1 and False:
                plt.figure()
                y_original =  self.y_augmented[0:sz_before_da-1]
                y_aug = self.y_augmented[sz_before_da:-1]
                plt.scatter(y_original, y_aug, s = 1)
                plt.xlabel('original yield')
                plt.ylabel('noise added')
                plt.show()
                print('debug')

        # adjust dimension of lenTS
        first = (cst.first_month_input_local_year) * 3
        if self.D == 1:
            return self.X_augmented[:,first:first+lenTS,:], self.region_ohe_augmented, self.y_augmented
        elif self.D == 2:
            return self.X_augmented[:,:,first:first+lenTS,:], self.region_ohe_augmented, self.y_augmented


