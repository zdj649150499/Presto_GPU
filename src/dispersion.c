#include "presto.h"


int thread_omp;


void gettread( int thread_input)
{
    thread_omp = thread_input;
}



double tree_max_dm(int numchan, double dt, double lofreq, double hifreq)
/* Return the maximum Dispersion Measure (dm) in cm-3 pc, the  */
/* tree de-dispersion technique can correct for given a sample */
/* interval 'dt', the number of channels 'numchan', and the    */
/* low and high observation frequencies in MHz.                */
{
    if (lofreq == 0.0 || hifreq == 0.0)
        return 0.0;
    else
        return 0.000241 * (numchan - 1) * dt /
            ((1.0 / (lofreq * lofreq)) - (1.0 / (hifreq * hifreq)));
}


double smearing_from_bw(double dm, double center_freq, double bandwidth)
/* Return the smearing in seconds caused by dispersion, given  */
/* a Dispersion Measure (dm) in cm-3 pc, the central frequency */
/* and the bandwith of the observation in MHz.                 */
{
    if (center_freq == 0.0)
        return 0.0;
    else
        return dm * bandwidth / (0.0001205 * center_freq * center_freq *
                                 center_freq);
}


double delay_from_dm(double dm, double freq_emitted)
/* Return the delay in seconds caused by dispersion, given  */
/* a Dispersion Measure (dm) in cm-3 pc, and the emitted    */
/* frequency (freq_emitted) of the pulsar in MHz.           */
{
    if (freq_emitted == 0.0)
        return 0.0;
    else
        return dm / (0.000241 * freq_emitted * freq_emitted);
}


double dm_from_delay(double delay, double freq_emitted)
/* Return the Dispersion Measure in cm-3 pc, that would     */
/* cause a pulse emitted at frequency 'freq_emitted' to be  */
/* delayed by 'delay' seconds.                              */
{
    if (freq_emitted == 0.0)
        return 0.0;
    else
        return delay * 0.000241 * freq_emitted * freq_emitted;
}


double *dedisp_delays(int numchan, double dm, double lofreq,
                      double chanwidth, double voverc)
/* Return an array of delays (sec) for dedispersing 'numchan'    */
/* channels at a DM of 'dm'.  'lofreq' is the center frequency   */
/* in MHz of the lowest frequency channel.  'chanwidth' is the   */
/* width in MHz of each channel.  'voverc' is the observatory's  */
/* velocity towards or away from the source.  This is to adjust  */
/* the frequencies for doppler effects (for no correction use    */
/* voverc=0).  The returned array is allocated by this routine.  */
{
    int ii;
    double *delays, freq;

    delays = gen_dvect(numchan);
    for (ii = 0; ii < numchan; ii++) {
        freq = doppler(lofreq + ii * chanwidth, voverc);
        delays[ii] = delay_from_dm(dm, freq);
    }
    return delays;
}

void *dedisp_delays_1(int numchan, double dm, double lofreq,
                      double chanwidth, double voverc, double *delays)
/* Return an array of delays (sec) for dedispersing 'numchan'    */
/* channels at a DM of 'dm'.  'lofreq' is the center frequency   */
/* in MHz of the lowest frequency channel.  'chanwidth' is the   */
/* width in MHz of each channel.  'voverc' is the observatory's  */
/* velocity towards or away from the source.  This is to adjust  */
/* the frequencies for doppler effects (for no correction use    */
/* voverc=0).  The returned array is allocated by this routine.  */
{
    int ii;
    double freq;

    // delays = gen_dvect(numchan);
    for (ii = 0; ii < numchan; ii++) {
        freq = doppler(lofreq + ii * chanwidth, voverc);
        delays[ii] = delay_from_dm(dm, freq);
    }
    // return delays;
}


void dedisp(unsigned char *data, unsigned char *lastdata, int numpts,
            int numchan, double *delays, float *result)
/* De-disperse a stretch of data with numpts * numchan points. */
/* The delays (in bins) are in dispdelays for each channel.    */
/* The result is returned in result.  The input data and       */
/* dispdelays are always in ascending frequency order.         */
/* Input data are ordered in time, with the channels stored    */
/* together at each time point.                                */
{
    int ii, jj, kk, ll;

    /* Initialize the result array */
    for (ii = 0; ii < numpts; ii++)
        result[ii] = 0.0;

    /* De-disperse */
    for (ii = 0; ii < numchan; ii++) {
        jj = ii + delays[ii] * numchan;
        for (kk = 0; kk < numpts - delays[ii]; kk++, jj += numchan)
            result[kk] += lastdata[jj];
        jj = ii;
        for (ll = 0; kk < numpts; kk++, jj += numchan, ll++)
            result[kk] += data[jj];
    }
}


double *subband_delays(int numchan, int numsubbands, double dm,
                       double lofreq, double chanwidth, double voverc)
/* Return an array of delays (sec) for the highest frequency  */
/* channels of each subband used in a subband de-dispersion.  */
/* These are the delays described in the 'Note:' in the       */
/* description of subband_search_delays().  See the comments  */
/* for dedisp_delays() for more info.                         */
{
    int chan_per_subband;
    double subbandwidth, losub_hifreq;

    chan_per_subband = numchan / numsubbands;
    subbandwidth = chanwidth * chan_per_subband;
    losub_hifreq = lofreq + subbandwidth - chanwidth;

    /* Calculate the appropriate delays to subtract from each subband */

    return dedisp_delays(numsubbands, dm, losub_hifreq, subbandwidth, voverc);
}


void *subband_delays_1(int numchan, int numsubbands, double dm,
                       double lofreq, double chanwidth, double voverc, double *subbanddelays)
/* Return an array of delays (sec) for the highest frequency  */
/* channels of each subband used in a subband de-dispersion.  */
/* These are the delays described in the 'Note:' in the       */
/* description of subband_search_delays().  See the comments  */
/* for dedisp_delays() for more info.                         */
{
    int chan_per_subband;
    double subbandwidth, losub_hifreq;

    chan_per_subband = numchan / numsubbands;
    subbandwidth = chanwidth * chan_per_subband;
    losub_hifreq = lofreq + subbandwidth - chanwidth;

    /* Calculate the appropriate delays to subtract from each subband */

    dedisp_delays_1(numsubbands, dm, losub_hifreq, subbandwidth, voverc, subbanddelays);
}


double *subband_search_delays(int numchan, int numsubbands, double dm,
                              double lofreq, double chanwidth, double voverc)
/* Return an array of delays (sec) for a subband DM search.  The      */
/* delays are calculated normally for each of the 'numchan' channels  */
/* using the appropriate frequencies at the 'dm'.  Then the delay     */
/* from the highest frequency channel of each of the 'numsubbands'    */
/* subbands is subtracted from each subband.  This gives the subbands */
/* the correct delays for each freq in the subband, but the subbands  */
/* themselves are offset as if they had not been de-dispersed.  This  */
/* way, we can call float_dedisp() on the subbands if needed.         */
/* 'lofreq' is the center frequency in MHz of the lowest frequency    */
/* channel.  'chanwidth' is the width in MHz of each channel.  The    */
/* returned array is allocated by this routine.  'voverc' is used to  */
/* correct the input frequencies for doppler effects.  See the        */
/* comments in dedisp_delays() for more info.                         */
/* Note:  When performing a subband search, the delays for each       */
/*   subband must be calculated with the frequency of the highest     */
/*   channel in each subband, _not_ the center subband frequency.     */
{
    int ii, jj, chan_per_subband;
    double *delays, *subbanddelays;

    chan_per_subband = numchan / numsubbands;

    /* Calculate the appropriate delays to subtract from each subband */

    subbanddelays = subband_delays(numchan, numsubbands, dm,
                                   lofreq, chanwidth, voverc);

    /* Calculate the appropriate delays for each channel */

    delays = dedisp_delays(numchan, dm, lofreq, chanwidth, voverc);
    for (ii = 0; ii < numsubbands; ii++)
        for (jj = 0; jj < chan_per_subband; jj++)
            delays[ii * chan_per_subband + jj] -= subbanddelays[ii];
    vect_free(subbanddelays);

    return delays;
}


void dedisp_subbands(float *data, float *lastdata,
                     int numpts, int numchan,
                     int *delays, int numsubbands, float *result, int blockN, int thisblock)
// De-disperse a stretch of data with numpts * numchan points into
// numsubbands subbands.  Each time point for each subband is a float
// in the result array.  The result array order is all the times for
// each subband, starting with lowest freq subband.  The delays (in
// bins) are in delays for each channel.  The input data and
// dispdelays are always in ascending frequency order.  Input data are
// contiguous channels with all of their points in time, starting with
// the lowest freq channel.
{
    const int chan_per_subband = numchan / numsubbands;
    long long ii, jj, kk, loffset;
    float *sub;

    /* Initialize the result array */
    // loffset = (long long)(numpts) * numsubbands;
    // for (ii = 0; ii < loffset; ii++)
    //     result[ii] = 0.0f;
    
    for (ii = 0; ii < numsubbands; ii++)
    {
        sub = result + (long long)(ii*numpts*blockN + (thisblock-1) * numpts);
        for (jj = 0; jj < numpts; jj++)
            sub[jj] = 0.0f;
    }

    /* De-disperse into the subbands */
/* #ifdef _OPENMP */
/* #pragma omp parallel for schedule(static,chan_per_subband)\ */
/*    default(none) private(ii,jj) shared(result,data,lastdata,delays,numchan,numpts) */
/* #endif */
    for (ii = 0; ii < numchan; ii++) {
        const int subnum = ii / chan_per_subband;
        const int dind = delays[ii];
        // float *sub = result + subnum * numpts;
        sub = result + subnum * numpts * blockN + (thisblock-1) * numpts;
        const long long loffset = ii * numpts;
        float *chan = lastdata + loffset + dind;
// #ifdef _OPENMP
// #pragma omp parallel for private(jj) shared(sub,chan,numpts)
// #endif
        for (jj = 0; jj < numpts - dind; jj++)
            sub[jj] += chan[jj];
        chan = data + ii * numpts;
        for (jj = numpts - dind, kk = 0; jj < numpts; jj++, kk++)
            sub[jj] += chan[kk];
    }
}

void dedisp_subbands_cache(unsigned char *data, float *data_scl, float *data_offs, unsigned char *lastdata, float *lastdata_scl, float *lastdata_offs,
                     int numpts, int numchan,
                     int *delays, int numsubbands, float *result, int blockN, int thisblock)
// De-disperse a stretch of data with numpts * numchan points into
// numsubbands subbands.  Each time point for each subband is a float
// in the result array.  The result array order is all the times for
// each subband, starting with lowest freq subband.  The delays (in
// bins) are in delays for each channel.  The input data and
// dispdelays are always in ascending frequency order.  Input data are
// contiguous channels with all of their points in time, starting with
// the lowest freq channel.
{
    const int chan_per_subband = numchan / numsubbands;
    long long ii, jj, kk, loffset;
    float *sub;

    /* Initialize the result array */
    // loffset = (long long)(numpts) * numsubbands;
    // for (ii = 0; ii < loffset; ii++)
    //     result[ii] = 0.0f;
    
    for (ii = 0; ii < numsubbands; ii++)
    {
        sub = result + (long long)(ii*numpts*blockN + (thisblock-1) * numpts);
        for (jj = 0; jj < numpts; jj++)
            sub[jj] = 0.0f;
    }

    /* De-disperse into the subbands */
/* #ifdef _OPENMP */
/* #pragma omp parallel for schedule(static,chan_per_subband)\ */
/*    default(none) private(ii,jj) shared(result,data,lastdata,delays,numchan,numpts) */
/* #endif */
    for (ii = 0; ii < numchan; ii++) {
        const int subnum = ii / chan_per_subband;
        const int dind = delays[ii];
        // float *sub = result + subnum * numpts;
        sub = result + subnum * numpts * blockN + (thisblock-1) * numpts;
        const long long loffset = ii * numpts;
        unsigned char *chan = lastdata + loffset + dind;
// #ifdef _OPENMP
// #pragma omp parallel for private(jj) shared(sub,chan,numpts, lastdata_scl, lastdata_offs, data_scl, data_offs, lastdata, data, ii, dind)
// #endif
        for (jj = 0; jj < numpts - dind; jj++)
            sub[jj] += (lastdata_offs[ii] + chan[jj]*lastdata_scl[ii]);
        chan = data + ii * numpts;
        for (jj = numpts - dind, kk = 0; jj < numpts; jj++, kk++)
            sub[jj] += (data_offs[ii] + chan[kk]*data_scl[ii]);
    }
}



void float_dedisp(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result, int transpose)
// De-disperse a stretch of data with numpts * numchan points. The
// delays (in bins) are in delays for each channel.  The result is
// returned in result.  The input data and delays are always in
// ascending frequency order.  Input data are ordered in time, with
// the channels stored together at each time point.
{
    int ii, jj, kk;

    // if(approx_mean != 0.0f)
        for (ii = 0; ii < numpts; ii++)
            result[ii] = -approx_mean;

    /* De-disperse */
    if(transpose)
    {
        /*time first*/
        int delays_tmp;

        for (ii = 0; ii < numchan; ii++) {
            delays_tmp = delays[ii];
            jj = ii*numpts + delays_tmp;
            for (kk = 0; kk < numpts - delays_tmp; kk++, jj++)
                result[kk] += lastdata[jj];
            jj = ii*numpts;
            for (; kk < numpts; kk++, jj++)
                result[kk] += data[jj];
        }
    }
    else
    {
        /*freq first*/
        for (ii = 0; ii < numchan; ii++) {
            jj = ii + delays[ii] * numchan;
            for (kk = 0; kk < numpts - delays[ii]; kk++, jj += numchan)
                result[kk] += lastdata[jj];
            jj = ii;
            for (; kk < numpts; kk++, jj += numchan)
                result[kk] += data[jj];
        }
    }
}

void float_dedisp_time(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result)
// De-disperse a stretch of data with numpts * numchan points. The
// delays (in bins) are in delays for each channel.  The result is
// returned in result.  The input data and delays are always in
// ascending frequency order.  Input data are ordered in time, with
// the channels stored together at each time point.
{
    int ii, jj, kk;

    // if(approx_mean != 0.0f)
        for (ii = 0; ii < numpts; ii++)
            result[ii] = -approx_mean;

    /* De-disperse */
    /*time first*/
    int delays_tmp;

    for (ii = 0; ii < numchan; ii++) {
        delays_tmp = delays[ii];
        jj = ii*numpts + delays_tmp;
        for (kk = 0; kk < numpts - delays_tmp; kk++, jj++)
            result[kk] += lastdata[jj];
        jj = ii*numpts;
        for (; kk < numpts; kk++, jj++)
            result[kk] += data[jj];
    }

}


void combine_subbands(double *inprofs, foldstats * stats,
                      int numparts, int numsubbands, int proflen,
                      int *delays, double *outprofs, foldstats * outprofstats)
/* Combine 'nparts' sets of 'numsubbands' profiles, each of length     */
/* 'proflen' into a 'nparts' de-dispersed profiles.  The de-dispersion */
/* uses the 'delays' (of which there are 'numsubbands' many) to        */
/* show how many bins to shift each profile to the right.  Only        */
/* positive numbers may be used (left shifts may be accomplished using */
/* the shift modulo 'proflen').  The 'stats' about the profiles are    */
/* combined as well and the combined stats are returned in             */
/* 'outprofstats'. All arrays must be pre-allocated.                   */
{
    int ii, jj, kk, ptsperpart;
    int partindex, profindex, ptindex, outprofindex, statindex;

    /* Set the output profiles and statistics to 0.0 */

    for (ii = 0; ii < numparts * proflen; ii++)
        outprofs[ii] = 0.0;
    for (ii = 0; ii < numparts; ii++) {
        // initialize_foldstats(&(outprofstats[ii]));
        outprofstats[ii].numdata = 0.0;       /* Number of data bins folded         */
        outprofstats[ii].data_avg = 0.0;      /* Average level of the data bins     */
        outprofstats[ii].data_var = 0.0;      /* Variance of the data bins          */
        outprofstats[ii].numprof = 0.0;       /* Number of bins in the profile      */
        outprofstats[ii].prof_avg = 0.0;      /* Average level of the profile bins  */
        outprofstats[ii].prof_var = 0.0;      /* Variance of the profile bins       */
        outprofstats[ii].redchi = 0.0;        /* Reduced chi-squared of the profile */
        outprofstats[ii].numprof = stats[0].numprof;
    }
    ptsperpart = numsubbands * proflen;

    /* Combine the profiles */

    for (ii = 0; ii < numparts; ii++) { /* Step through parts */
        outprofindex = ii * proflen;
        partindex = ii * ptsperpart;
        statindex = ii * numsubbands;
        outprofstats[ii].numdata += stats[statindex].numdata;
        for (jj = 0; jj < numsubbands; jj++) {  /* Step through subbands */
            profindex = partindex + jj * proflen;

            /* low part of profile  */

            ptindex = profindex + delays[jj];
            for (kk = 0; kk < proflen - delays[jj]; kk++, ptindex++)
                outprofs[outprofindex + kk] += inprofs[ptindex];

            /* high part of profile */

            ptindex = profindex;
            for (; kk < proflen; kk++, ptindex++)
                outprofs[outprofindex + kk] += inprofs[ptindex];

            /* Update the foldstats */

            outprofstats[ii].data_avg += stats[statindex + jj].data_avg;
            outprofstats[ii].data_var += stats[statindex + jj].data_var;
            outprofstats[ii].prof_avg += stats[statindex + jj].prof_avg;
            outprofstats[ii].prof_var += stats[statindex + jj].prof_var;
        }
    }
}


void combine_subbands_1(double *inprofs,
                      int numparts, int numsubbands, int proflen,
                      int *delays, double *outprofs)
/* Combine 'nparts' sets of 'numsubbands' profiles, each of length     */
/* 'proflen' into a 'nparts' de-dispersed profiles.  The de-dispersion */
/* uses the 'delays' (of which there are 'numsubbands' many) to        */
/* show how many bins to shift each profile to the right.  Only        */
/* positive numbers may be used (left shifts may be accomplished using */
/* the shift modulo 'proflen').  The 'stats' about the profiles are    */
/* combined as well and the combined stats are returned in             */
/* 'outprofstats'. All arrays must be pre-allocated.                   */
{
    int ii, jj, kk, ptsperpart;
    int partindex, profindex, ptindex, outprofindex;

    /* Set the output profiles and statistics to 0.0 */

    for (ii = 0; ii < numparts * proflen; ii++)
        outprofs[ii] = 0.0;

    ptsperpart = numsubbands * proflen;

    /* Combine the profiles */

    for (ii = 0; ii < numparts; ii++) { /* Step through parts */
        outprofindex = ii * proflen;
        partindex = ii * ptsperpart;
        for (jj = 0; jj < numsubbands; jj++) {  /* Step through subbands */
            profindex = partindex + jj * proflen;

            /* low part of profile  */

            ptindex = profindex + delays[jj];
            for (kk = 0; kk < proflen - delays[jj]; kk++, ptindex++)
                outprofs[outprofindex + kk] += inprofs[ptindex];

            /* high part of profile */

            ptindex = profindex;
            for (; kk < proflen; kk++, ptindex++)
                outprofs[outprofindex + kk] += inprofs[ptindex];
        }
    }
}

