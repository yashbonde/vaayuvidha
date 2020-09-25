# Data-Spec

## Inputs

There are 127 weather stations, each capturing the following data points:
```
'prcp': Amount of precipitation in millimetres (last hour)
'stp' : Air pressure for the hour in hPa to tenths (instant)
'gbrd': Solar radiation KJ/m2
'temp': Air temperature (instant) in celsius degrees
'dewp': Dew point temperature (instant) in celsius degrees
'hmdy': Relative humid in % (instant)
'wdsp': Wind speed in metres per second
'wdct': Wind direction in radius degrees (0-360)
'gust': Wind gust in metres per second
```

For each station location metrics:
```
'elvt': Elevation
'lat' : Latitude
'lon' : Longitude
```

* At any given time "t" and station "x", `feat_x_t = [9,] + [3,] = [F,]`
* At any given time "t" and for "N" stations, `feat_t = [N, F,]`
* For usage, you also need past information, for "T" time steps (context-size), for "N" stations, `input = [T, N, F,]`.
* What model does, `[T, N, F,]` returns information for T+1hr for N stations each having F features/metrics.

Go part by part:
* `nodeEncoder`: if I don't have data for some station, I can mask it using, `ws_mask_t`. Now all I care is, number of samples is low.
* `temporal`: if I don't have data for any station at a time point "t", I can mask it using `temporal_mask`
* `nodeDecoder`: if I don't have data for some station at time `t+1`, I need a `future_ws_mask_t_plus_1`, this will allow me to handle loss.

```
NETWORK
=======

nodeEncoder: [N, F] --> [N, E] --> MessagePassing --> [N, E] + [1, G]

If I have to do it over multiple time steps (T / transformer.context_size)

nodeEncoder: [T, N, F] --> [T, N, E] --> MessagePassing --> [T, N, E] + [T, G]

Raigarh mai garmi hai, uska "yeh" effect Korba mai pada. Raipur pe badal hai, Bilaspur mai garmi.

Raigarh <--> Korba <--> Raipur <--> Bilaspur (on 3rd Day)

----

temporal: [T, G] --> [casual_attention] --> [T, G]

If I have data of changes in teh graph over multiple time steps, I can train what happens next.
Like any GPT. (graph_at_t --> graph_at_t_plus_1)

Char din se CG, garmi badh rahi hai, 5th day pe Garmi kam hogi.

----

nodeDecoder: [N, F] --> [N, E] --> MessagePassing --> [N, E] + [1, G]

If I have to do it over multiple time steps (T / transformer.context_size)

nodeDecoder: [T, N, F] --> [T, N, E] --> MessagePassing --> [T, N, E] + [T, G]

Raigarh <--> Korba <--> Raipur <--> Bilaspur (predict for 4th Day)

```

### Example
```
-- Uncorrupted case
* for this duration (2000-05-07-12:00 .... 2000-05-19-18:00)
* go to each station and collect "F" features --> [T,N,F]

-- Corrupted case
* for this duration (2000-05-07-12:00 .... 2000-05-19-18:00)
* go to each date
    * if "n" ws were not gathering data
        * if n < 10% of all the cities --> valid data point
    * go to "n" and collect "F" features --> [,N,F]
        * if "f" not available fill with mean of +-3 timesteps
```

### Caveats

There is more data available today then was ever in the past, all stations were not started simultaneously.
Some started 1 year ago, some in 2000. But we cannot discard the entire data.

```
1. parsing the data --> downloading and running the scripts
2. create a dummy train file and test file --> this is what the Dataset Object reads
3. Return the tesors

----

4. go back to data --> identify useful samples
5. Break into test samples
6. -9999 samples (human-validation)
7. write the files --> goto (2)
```

