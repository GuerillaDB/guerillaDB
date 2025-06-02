-----------------------------------------------------------------------------------------------------------------
---------------Procedure checking for missing price datapoints given max timestep equal to parameter-------------
---------------Note the DATEDIFF(SECOND, ...) rounds the actual timestamp difference to full seconds-------------
-----------------------------------------------------------------------------------------------------------------

CREATE PROCEDURE dbo.ValidateTimestep
AS
BEGIN
    DECLARE @max_timestep INT;
    
    -- Get parameter from validation table
    SELECT @max_timestep = CAST(Parameter1_Value AS INT)
    FROM ValidationProceduresParameters 
    WHERE ProcedureName = 'ValidateTimestep';

    WITH TimeDiffs AS (
        SELECT
            Timestamp,
            DATEDIFF(SECOND, LAG(Timestamp, 1, Timestamp) OVER (ORDER BY Timestamp), Timestamp) AS TimeDiff
        FROM
            dbo.KibotUpsampledBatch
    )
    SELECT
        Timestamp
    FROM
        TimeDiffs
    WHERE
        TimeDiff > @max_timestep;
END