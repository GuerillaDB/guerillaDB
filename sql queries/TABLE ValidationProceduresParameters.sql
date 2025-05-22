-- Table creation
CREATE TABLE ValidationProceduresParameters (
    ProcedureName VARCHAR(250),
    ProcedureDescription VARCHAR(500),
    
    Parameter1_Value DECIMAL(18,4),
    Parameter1_Name VARCHAR(100),
    Parameter1_Description VARCHAR(500),
    
    Parameter2_Value DECIMAL(18,4), 
    Parameter2_Name VARCHAR(100),
    Parameter2_Description VARCHAR(500),
    
    Parameter3_Value DECIMAL(18,4),
    Parameter3_Name VARCHAR(100),
    Parameter3_Description VARCHAR(500),
    
    Parameter4_Value DECIMAL(18,4),
    Parameter4_Name VARCHAR(100),
    Parameter4_Description VARCHAR(500),
    
    Parameter5_Value DECIMAL(18,4),
    Parameter5_Name VARCHAR(100),
    Parameter5_Description VARCHAR(500),
    
    Parameter6_Value DECIMAL(18,4),
    Parameter6_Name VARCHAR(100),
    Parameter6_Description VARCHAR(500),
    
    Parameter7_Value DECIMAL(18,4),
    Parameter7_Name VARCHAR(100),
    Parameter7_Description VARCHAR(500),
    
    Parameter8_Value DECIMAL(18,4),
    Parameter8_Name VARCHAR(100),
    Parameter8_Description VARCHAR(500)
)
;
-- Insert statement
INSERT INTO ValidationProceduresParameters (
    ProcedureName,
    ProcedureDescription,
    Parameter1_Value, Parameter1_Name, Parameter1_Description,
    Parameter2_Value, Parameter2_Name, Parameter2_Description,
    Parameter3_Value, Parameter3_Name, Parameter3_Description,
    Parameter4_Value, Parameter4_Name, Parameter4_Description,
    Parameter5_Value, Parameter5_Name, Parameter5_Description
)
VALUES (
    'ValidateTradeUpsampled',
    'Bad tick valuation for table KibotUpsampledBatch, column Trade',
    60, 'sampling_rate', 'interval between rolling windows in seconds',
    0.001, 'consecutive_change_threshold', 'minimum change between consecutive values to flag as bad tick',
    0.0001, 'window_change_threshold', 'minimum change between the extreme value of the time window and final value of the window',
    0.2, 'price_gap_threshold', 'fraction of the difference between final window value and extreme, values outside of which are considered trailing values',
    30, 'num_trailing_values', 'minimum number of trailing values to consider the extreme a valid datapoint'
)
;
INSERT INTO ValidationProceduresParameters (
    ProcedureName,
    ProcedureDescription,
    Parameter1_Value, Parameter1_Name, Parameter1_Description
)
VALUES (
    'ValidateTimestep',
    'Timestep validation for table KibotUpsampledBatch, column Timestamp',
    1, 'max_diff_seconds', 'max allowed timedelta in seconds between two consecutive timestamps',
)