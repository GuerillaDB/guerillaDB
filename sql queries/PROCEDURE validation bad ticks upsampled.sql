CREATE PROCEDURE ValidateTradeUpsampled
AS 
BEGIN 

DECLARE @sampling_rate INT;
DECLARE @consecutive_change_threshold DECIMAL(10,4);
DECLARE @window_change_threshold DECIMAL(10,4);
DECLARE @price_gap_threshold DECIMAL(10,4);
DECLARE @num_trailing_values INT;

SELECT 
    @sampling_rate = CAST(Parameter1_Value AS INT),
    @consecutive_change_threshold = CAST(Parameter2_Value AS DECIMAL(10,4)),
    @window_change_threshold = CAST(Parameter3_Value AS DECIMAL(10,4)),
    @price_gap_threshold = CAST(Parameter4_Value AS DECIMAL(10,4)),
    @num_trailing_values = CAST(Parameter5_Value AS INT)
FROM ValidationProceduresParameters 
WHERE ProcedureName = 'ValidateTradeUpsampled';


WITH MinMaxTimes AS (
    SELECT
        MIN(Timestamp) AS MinTime,
        MAX(Timestamp) AS MaxTime,
        DATEDIFF(SECOND, MIN(Timestamp), MAX(Timestamp)) AS TotalSeconds
    FROM
        dbo.KibotUpsampledBatch
),
GeneratedTimes AS (
    SELECT
        DATEADD(SECOND, n.Number, m.MinTime) AS GeneratedTimestamp
    FROM
        Numbers n
    CROSS JOIN
        MinMaxTimes m
    WHERE
        n.Number <= m.TotalSeconds AND n.Number >= 0
),
FilledData AS (
    SELECT
        gt.GeneratedTimestamp AS PriceTimestamp,
        d.Trade,
        ROW_NUMBER() OVER (ORDER BY gt.GeneratedTimestamp) AS RowNumber
    FROM
        GeneratedTimes gt
    LEFT JOIN
        dbo.KibotUpsampledBatch d ON gt.GeneratedTimestamp = d.Timestamp
),
LastNonNullTimestamp AS (
    SELECT
        PriceTimestamp,
        Trade,
        MAX(CASE WHEN Trade IS NOT NULL THEN PriceTimestamp END) OVER (ORDER BY PriceTimestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS LastNonNullTs
    FROM
        FilledData
),
FilledTradeValues AS (
    SELECT
        PriceTimestamp,
        FilledTradeValue AS FilledTrade,
        ROW_NUMBER() OVER (ORDER BY PriceTimestamp) AS rn,
        LAG(FilledTradeValue, 1, FilledTradeValue) OVER (ORDER BY PriceTimestamp) AS PrevFilledTrade,
        CASE
            WHEN LAG(FilledTradeValue, 1, FilledTradeValue) OVER (ORDER BY PriceTimestamp) = 0 THEN 0 -- Avoid division by zero
            ELSE (FilledTradeValue - LAG(FilledTradeValue, 1, FilledTradeValue) OVER (ORDER BY PriceTimestamp)) / ABS(LAG(FilledTradeValue, 1, FilledTradeValue) OVER (ORDER BY PriceTimestamp))
        END AS PctChange
    FROM (
        SELECT
            lnt.PriceTimestamp,
            FIRST_VALUE(lnt.Trade) OVER (PARTITION BY lnt.LastNonNullTs ORDER BY lnt.PriceTimestamp) AS FilledTradeValue,
            ROW_NUMBER() OVER (ORDER BY lnt.PriceTimestamp) AS rn
        FROM
            LastNonNullTimestamp lnt
        WHERE lnt.LastNonNullTs IS NOT NULL
    ) AS FilledTradeBase
),
SampledTimestamps AS (
    SELECT
        PriceTimestamp,
        rn AS SampledRn
    FROM
        FilledTradeValues
    WHERE
        rn >= (5 *  @sampling_rate) AND rn %  @sampling_rate = 0  --parameter dependent
),
FullDataWithRollingMinMax AS (
    SELECT
        PriceTimestamp,
        FilledTrade,
        rn,
        PctChange,
        MAX(FilledTrade) OVER (ORDER BY rn ROWS BETWEEN 299 PRECEDING AND CURRENT ROW) AS FullMaxTradeLast5Min,  --hardcoded parameter dependent
        MIN(FilledTrade) OVER (ORDER BY rn ROWS BETWEEN 299 PRECEDING AND CURRENT ROW) AS FullMinTradeLast5Min,  --hardcoded parameter dependent
        MAX(PctChange) OVER (ORDER BY rn ROWS BETWEEN 299 PRECEDING AND CURRENT ROW) AS MaxPercChangeLast5Min,  --hardcoded parameter dependent
        MIN(PctChange) OVER (ORDER BY rn ROWS BETWEEN 299 PRECEDING AND CURRENT ROW) AS MinPercChangeLast5Min  --hardcoded parameter dependent
    FROM
        FilledTradeValues
),
MaxMinForSampled AS (
    SELECT
        st.PriceTimestamp,
        fwr.FilledTrade,
        fwr.rn,
        fwr.PctChange,
        fwr.FullMaxTradeLast5Min AS MaxTradeLast5Min,
        fwr.FullMinTradeLast5Min AS MinTradeLast5Min,
        fwr.MaxPercChangeLast5Min,
        fwr.MinPercChangeLast5Min
    FROM
        SampledTimestamps st
    JOIN
        FullDataWithRollingMinMax fwr ON st.SampledRn = fwr.rn
),
significant_change_window AS (
    SELECT
        PriceTimestamp,
        FilledTrade,
        rn AS SampledRn,
        MaxTradeLast5Min,
        MinTradeLast5Min
    FROM
        MaxMinForSampled
    WHERE
        MaxPercChangeLast5Min > @consecutive_change_threshold OR MinPercChangeLast5Min < -@consecutive_change_threshold  --parameter dependent
    -- Checking for a significant positive or negative jump within the window
),
max_above_threshold AS (
    SELECT
        scw.PriceTimestamp,
        scw.FilledTrade,
        scw.SampledRn,
        scw.MaxTradeLast5Min
    FROM
        significant_change_window scw
    WHERE
        scw.MaxTradeLast5Min > scw.FilledTrade * (1 + @window_change_threshold)  --parameter dependent
    -- Filtering out cases where the rolling maximum isn't significantly higher
    -- than the current trade price within the last 5 minutes (that had a significant jump).
),
trailing_values_above AS (
    SELECT
        mat.PriceTimestamp,
        mat.FilledTrade,
        mat.MaxTradeLast5Min,
        COUNT(ftv_trailing.FilledTrade) AS CountTrailingAbove
    FROM
        max_above_threshold mat
    INNER JOIN
        FilledTradeValues ftv_trailing ON ftv_trailing.rn >= mat.SampledRn - 299  --hardcoded parameter dependent
                                         AND ftv_trailing.rn < mat.SampledRn
                                         AND ftv_trailing.FilledTrade BETWEEN 
											(mat.FilledTrade + @price_gap_threshold * (mat.MaxTradeLast5Min - mat.FilledTrade)) 
											AND mat.MaxTradeLast5Min  --parameter dependent
    GROUP BY
        mat.PriceTimestamp,
        mat.FilledTrade,
        mat.MaxTradeLast5Min
),
min_below_threshold AS (
    SELECT
        scw.PriceTimestamp,
        scw.FilledTrade,
        scw.SampledRn,
        scw.MinTradeLast5Min
    FROM
        significant_change_window scw
    WHERE
        scw.MinTradeLast5Min < scw.FilledTrade * (1 - @window_change_threshold)  --parameter dependent
    -- Filtering out cases where the rolling minimum isn't significantly lower
    -- than the current trade price within the last 5 minutes (that had a significant jump).
),
trailing_values_below AS (
    SELECT
        mbt.PriceTimestamp,
        mbt.FilledTrade,
        mbt.MinTradeLast5Min,
        COUNT(ftv_trailing.FilledTrade) AS CountTrailingBelow
    FROM
        min_below_threshold mbt
    INNER JOIN
        FilledTradeValues ftv_trailing ON ftv_trailing.rn >= mbt.SampledRn - 299  --parameter dependent
                                         AND ftv_trailing.rn < mbt.SampledRn
                                         AND ftv_trailing.FilledTrade BETWEEN mbt.MinTradeLast5Min AND 
											(mbt.FilledTrade - @price_gap_threshold * (mbt.FilledTrade - mbt.MinTradeLast5Min))  --parameter dependent
    GROUP BY
        mbt.PriceTimestamp,
        mbt.FilledTrade,
        mbt.MinTradeLast5Min
),
flagged_windows AS (
    SELECT
        PriceTimestamp,
        FilledTrade,
        ExtremeValue,
        TrailingObservations,
        rn,
        MaxTradeLast5Min,
        MinTradeLast5Min
    FROM (
        SELECT
            tva.PriceTimestamp,
            tva.FilledTrade,
            tva.MaxTradeLast5Min AS ExtremeValue,
            tva.CountTrailingAbove AS TrailingObservations,
            fwr_above.rn,
            fwr_above.FullMaxTradeLast5Min AS MaxTradeLast5Min,
            NULL AS MinTradeLast5Min
        FROM
            trailing_values_above tva
        INNER JOIN
            FullDataWithRollingMinMax fwr_above ON tva.PriceTimestamp = fwr_above.PriceTimestamp
        UNION ALL
        SELECT
            tvb.PriceTimestamp,
            tvb.FilledTrade,
            tvb.MinTradeLast5Min AS ExtremeValue,
            tvb.CountTrailingBelow AS TrailingObservations,
            fwr_below.rn,
            NULL AS MaxTradeLast5Min,
            fwr_below.FullMinTradeLast5Min AS MinTradeLast5Min
        FROM
            trailing_values_below tvb
        INNER JOIN
            FullDataWithRollingMinMax fwr_below ON tvb.PriceTimestamp = fwr_below.PriceTimestamp
    ) AS CombinedResults
    WHERE TrailingObservations < @num_trailing_values  --parameter dependent
)
SELECT
    fw.PriceTimestamp,
    fw.FilledTrade,
    fw.ExtremeValue,
    (
        SELECT TOP 1 ftv.PriceTimestamp
        FROM FilledTradeValues ftv
        WHERE ftv.rn BETWEEN fw.rn - 299 AND fw.rn - 1  --hardcoded parameter dependent (window size, not the 1)
        ORDER BY
            CASE
                WHEN fw.ExtremeValue = fw.MaxTradeLast5Min THEN ftv.FilledTrade
            END DESC,
            CASE
                WHEN fw.ExtremeValue = fw.MinTradeLast5Min THEN ftv.FilledTrade
            END ASC,
            ftv.rn ASC
    ) AS ExtremeTimestamp,
    fw.TrailingObservations
FROM
    flagged_windows fw
ORDER BY
    fw.PriceTimestamp;

END