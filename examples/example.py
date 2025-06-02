# imports, loading the data to dataframe
import guerilla_pack as guerilla
import pandas as pd


input_csv = r'file_to_csv.csv'

sql_server=r'SQL_SERVER_NAME'
sql_database='SQL_DATABASE_NAME'

postgresg_server=r'PG_SERVER_NAME'
postgres_database='PG_DATABASE_NAME'

# initialize GuerillaCompression instance
gc = guerilla.GuerillaCompression(server=sql_server, database=sql_database)

# change the instance's properties: dbms, server, database, verbose_level, number of worker threads
gc.dbms = 'postgres'
gc.server = 'POSTGRES_SERVER_NAME'
gc.database = 'POSTGRES_DATABASE_NAME'
gc.verbose_level = 'FULL'
gc.num_workers = 4


PROCESSING_CHUNK_SIZE = 400_000
# compress the data and save to DB
# from filepath:
_ = gc.compress_dataframe(input_csv, 
                          timestamp_col="TIMESTAMP_COLUMN_NAME", 
                          output_format='db', 
                          data_name='DATA_NAME',
                          processing_chunk_size=PROCESSING_CHUNK_SIZE)

# from pandas dataframe in memory:
df = pd.read_csv(input_csv)
_ = gc.compress_dataframe(df, 
                          timestamp_col="TIMESTAMP_COLUMN_NAME", 
                          output_format='db', 
                          data_name='DATA_NAME',
                          processing_chunk_size=PROCESSING_CHUNK_SIZE)

# from database table:
_ = gc.compress_dataframe('TABLE_NAME', 
                          timestamp_col="TIMESTAMP_COLUMN_NAME", 
                          output_format='db', 
                          data_name='TARGET_DATA_NAME',
                          processing_chunk_size=PROCESSING_CHUNK_SIZE)

# reconstruct data from DB to return a pandas DataFrame:
decompressed_df = gc.decompress_chunked_data(data_name='TARGET_DATA_NAME')

# reconstruct data based on start and end time rather than full dataset
start_time = '2022-01-01'
end_time = '2024-06-30'
decompressed_df_partial = gc.decompress_chunked_data(data_name='TARGET_DATA_NAME', 
                                                     start_time=start_time, 
                                                     end_time=end_time)

# appending data to an existing table
_ = gc.append_compressed('BATCH_DATA_NAME', 'TARGET_DATA_NAME', chunk_size=25_000, new_chunk=True)

# including feature routines to cacluate market features on the raw data and store them in DB
feature_routines=['numerical', 'categorical']
_ = gc.append_compressed('BATCH_DATA_NAME', 'TARGET_DATA_NAME', chunk_size=25_000, new_chunk=True,
                         feature_routine=feature_routines)

# performing a lookup on market features to retrieve ids of chunk satisfying the conditions
numerical_conditions = {
    'volatility': {
        'Trade': {'p90': {'>': 0.12}}
    },
    'price_jump_skew': {
        'ratio_bid_change': {'p90': {'>': 0.45}},
        'ratio_bid_pressure': {'p50': {'>': 0.5}}
    }
}
chunks = gc.feature_lookup('TARGET_DATA_NAME',
    numerical_conditions=numerical_conditions,
    categorical_conditions=[],
    operator='AND'
)


# ValidationManager class
vm = guerilla.ValidationManager()

# quick rundown of validation procedures stored in SQL Server
vm.procedure_lookup(show_parameters=False)

# more detailed summary with parameter values
vm.procedure_lookup(show_parameters=True)

# update a validation procedure parameter via ValidationManager:
vm.procedure_update(procedure_name='PROCEDURE_NAME', 
                    parameter_name='PARAMETER_NAME', 
                    new_value='PARAMETER_VALUE')

# run validation procedureS
validation_results = vm.run_validation()

# run only a subset of procedures
validation_results = vm.run_validation(['PROCEDURE_NAME_1', 'PROCEDURE_NAME_2'])






