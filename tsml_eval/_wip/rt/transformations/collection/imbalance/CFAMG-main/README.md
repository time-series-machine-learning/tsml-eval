# Mitigating Data Imbalance in Time Series Classification Based on Counterfactual Minority Samples Augmentation

In this repository, we provide the original PyTorch implementation of CFAMG framework.

## Dataset

We conducted extensive experiments on 53 real-world benchmark datasets selected from the UCR and UEA time series classification repositories. For details, please refer to the following table：
|     Type     |             Dataset            | Dimensions |        Pos Class       | #Length |      | Training |        |      | Test |         |
|:------------:|:------------------------------:|:----------:|:----------------------:|:-------:|:----:|:--------:|:------:|:----:|:----:|:-------:|
|              |                                |            |                        |         | #Neg |   #Pos   |   IR   | #Neg | #Pos |    IR   |
|  Univariate  |              ACSF1             |      1     |            1           |   1460  |  90  |    10    |  9.00  |  90  |  10  |  9.00   |
| Multivariate |    ArticularyWordRecognition   |      9     |          25.0          |   144   |  264 |    11    | 24.00  |  288 |  12  |  24.00  |
|  Univariate  |               Car              |      1     |            3           |   577   |  49  |    11    |  4.45  |  41  |  19  |  2.16   |
|  Univariate  |            Computers           |      1     |            2           |   720   |  125 |    41    |  3.05  |  125 |  125 |  1.00   |
|  Univariate  |          DodgerLoopDay         |      1     |          5, 2          |   288   |  51  |    16    |  3.19  |  56  |  21  |  2.67   |
|  Univariate  |           Earthquakes          |      1     |            1           |   512   |  264 |    58    |  4.55  |  104 |  35  |  2.97   |
|  Univariate  |             ECG5000            |      1     |            1           |   140   |  208 |    97    |  2.14  | 1873 | 2627 |  0.71   |
|  Univariate  |          EthanolLevel          |      1     |            4           |   1751  |  378 |    126   |  3.00  |  376 |  124 |  3.03   |
|  Univariate  |             FaceAll            |      1     |           14           |   131   |  520 |    40    | 13.00  | 1658 |  32  |  51.81  |
|  Univariate  |            FacesUCR            |      1     |         11, 14         |   131   |  190 |    10    | 19.00  | 1940 |  110 |  17.64  |
|  Univariate  |           FiftyWords           |      1     | 41, 50, 49, 42, 25, 34 |   270   |  438 |    12    | 36.50  |  423 |  32  |  13.22  |
|  Univariate  |              Fish              |      1     |            4           |   463   |  154 |    21    |  7.33  |  146 |  29  |  5.03   |
|  Univariate  |       FreezerRegularTrain      |      1     |            2           |   301   |  75  |    25    |  3.00  | 1425 | 1425 |  1.00   |
|  Univariate  |         GestureMidAirD2        |      1     |         23, 26         |   360   |  192 |    16    | 12.00  |  120 |  10  |  12.00  |
|  Univariate  |         GestureMidAirD3        |      1     |          3, 13         |   360   |  192 |    16    | 12.00  |  120 |  10  |  12.00  |
|  Univariate  |         GesturePebbleZ1        |      1     |            2           |   455   |  112 |    20    |  5.60  |  147 |  25  |  5.88   |
|  Univariate  |         GesturePebbleZ2        |      1     |            2           |   455   |  123 |    23    |  5.35  |  136 |  22  |  6.18   |
|  Univariate  |            GunPoint            |      1     |            1           |   150   |  26  |    10    |  2.60  |  74  |  76  |  0.97   |
|  Univariate  |         GunPointAgeSpan        |      1     |            2           |   150   |  68  |    22    |  3.09  |  160 |  156 |  1.03   |
|  Univariate  |               Ham              |      1     |            1           |   431   |  57  |    17    |  3.35  |  54  |  51  |  1.06   |
| Multivariate |      HandMovementDirection     |     10     |        backward        |   400   |  120 |    40    |  3.00  |  59  |  15  |  3.93   |
|  Univariate  |             Haptics            |      1     |            1           |   1092  |  137 |    18    |  7.61  |  248 |  60  |  4.13   |
|  Univariate  |             Herring            |      1     |            2           |   512   |  39  |    10    |  3.90  |  38  |  26  |  1.46   |
|  Univariate  |           InlineSkate          |      1     |          1, 7          |   1882  |  80  |    20    |  4.00  |  446 |  104 |  4.29   |
|  Univariate  |        ItalyPowerDemand        |      1     |            2           |    24   |  34  |    11    |  3.09  |  513 |  516 |  0.99   |
|  Univariate  |           Lightning2           |      1     |           -1           |   637   |  40  |    20    |  2.00  |  33  |  28  |  1.18   |
|  Univariate  |             Mallat             |      1     |         5, 1, 3        |   1024  |  41  |    14    |  2.93  | 1459 |  886 |  1.65   |
|  Univariate  |          MedicalImages         |      1     |          8, 6          |    99   |  368 |    13    | 28.31  |  726 |  34  |  21.35  |
|  Univariate  |       MelbournePedestrian      |      1     |            9           |    24   | 1040 |    98    | 10.61  | 2129 |  190 |  11.21  |
|  Univariate  |   MiddlePhalanxOutlineCorrect  |      1     |            0           |    80   |  388 |    70    |  5.54  |  166 |  125 |  1.33   |
|  Univariate  |     MixedShapesRegularTrain    |      1     |            5           |   1024  |  400 |    100   |  4.00  | 2111 |  314 |  6.72   |
|  Univariate  |      MixedShapesSmallTrain     |      1     |            5           |   1024  |  80  |    20    |  4.00  | 2111 |  314 |  6.72   |
| Multivariate |             NATOPS             |     24     |           2.0          |    51   |  150 |    30    |  5.00  |  150 |  30  |  5.00   |
|  Univariate  |      PickupGestureWiimoteZ     |      1     |          10, 9         |   361   |  40  |    10    |  4.00  |  40  |  10  |  4.00   |
|  Univariate  |        PigAirwayPressure       |      1     |   52, 51, 50, 49, 48   |   2000  |  94  |    10    |  9.40  |  188 |  20  |  9.40   |
|  Univariate  |         PigArtPressure         |      1     |   52, 51, 50, 49, 48   |   2000  |  94  |    10    |  9.40  |  188 |  20  |  9.40   |
|  Univariate  | ProximalPhalanxOutlineAgeGroup |      1     |            1           |    80   |  328 |    72    |  4.56  |  188 |  17  |  11.06  |
|  Univariate  |        ProximalPhalanxTW       |      1     |            3           |    80   |  384 |    16    | 24.00  |  203 |   2  | 101.50  |
| Multivariate |          RacketSports          |      6     |  squash_backhandboast  |    30   |  117 |    34    |  3.44  |  118 |  34  |  3.47   |
|  Univariate  |      RefrigerationDevices      |      1     |            3           |   720   |  250 |    125   |  2.00  |  250 |  125 |  2.00   |
|  Univariate  |           ScreenType           |      1     |            3           |   720   |  250 |    125   |  2.00  |  250 |  125 |  2.00   |
| Multivariate |       SelfRegulationSCP2       |      7     |       positivity       |   1152  |  100 |    33    |  3.03  |  90  |  90  |  1.00   |
|  Univariate  |        SemgHandGenderCh2       |      1     |            2           |   1500  |  150 |    50    |  3.00  |  390 |  210 |  1.86   |
|  Univariate  |       SemgHandMovementCh2      |      1     |            6           |   1500  |  375 |    75    |  5.00  |  375 |  75  |  5.00   |
|  Univariate  |      ShakeGestureWiimoteZ      |      1     |          10, 9         |   385   |  40  |    10    |  4.00  |  40  |  10  |  4.00   |
|  Univariate  |     SmallKitchenAppliances     |      1     |            3           |   720   |  250 |    125   |  2.00  |  250 |  125 |  2.00   |
|  Univariate  |         SmoothSubspace         |      1     |            3           |    15   |  100 |    50    |  2.00  |  100 |  50  |  2.00   |
|  Univariate  |           Strawberry           |      1     |            1           |   235   |  394 |    73    |  5.40  |  238 |  132 |  1.80   |
|  Univariate  |              Trace             |      1     |            2           |   275   |  79  |    21    |  3.76  |  71  |  29  |  2.45   |
|  Univariate  |           TwoPatterns          |      1     |            2           |   128   |  763 |    237   |  3.22  | 2989 | 1011 |  2.96   |
|  Univariate  |              Wine              |      1     |            2           |   234   |  30  |    10    |  3.00  |  27  |  27  |  1.00   |
|  Univariate  |              Worms             |      1     |            5           |   900   |  164 |    17    |  9.65  |  69  |   8  |  8.62   |
|  Univariate  |          WormsTwoClass         |      1     |            1           |   900   |  105 |    25    |  4.20  |  44  |  33  |  1.33   |

## Usage

#### Requirements

The code was trained with `python 3.8`, `pytorch 1.13.1, `cuda 11.7`, and `cudnn 8.5.0`.

  ```shell
# create virtual environment
conda create --name CFAMG python=3.8

# activate environment
conda activate CFAMG

# Install dependencies
conda env create -f environment.yml

#### Run code

The UCR and UEA datasets can be accessed through the **tslearn** package, which can be installed as follows:

```shell
pip install tslearn

```python
from tslearn.datasets import UCR_UEA_datasets

To train and generate minority samples CFAMG framework on a dataset, run the following command:

```shell
python main.py --log_dir <model_save_path>  --save_freq <frequency_to_save_model_checkpoint>  --num_epochs <max_iters_training> --latent_size <latent_size>  --batch_size <batch_size>  --lr <learning rate>  --tensorboard --seed <seed>```

We choose classifiers from the **tsai library** as a unified benchmark, and tsai can be installed as follows

```shell
pip install tsai


## Directory Structure

The code directory structure is shown as follows:
```shell

CFAMG
├── main.py  # entry for model training
├── cfamg.py  # training, generating minority samples of CFAMG
├── data_preprocess.py  # dataset loading and preprocessing
├── model_utlis.py  # common utility functions
├── network.py  # network layer definition
