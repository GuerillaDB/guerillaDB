import pyodbc
from typing import List, Optional, Union

class ValidationManager:
    def __init__(self, server: str = r'localhost\SQLEXPRESS', database: str = 'GuerillaCompression'):
        self.server = server
        self.database = database
        self.conn_str = f'Driver=SQL Server;Server={server};Database={database};Trusted_Connection=yes;'
        
    def run_validation(self, validation_list: Optional[List[str]] = None) -> None:
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            
            if validation_list is None:
                cursor.execute("SELECT ProcedureName FROM ValidationProceduresParameters")
                validation_list = [row[0] for row in cursor.fetchall()]
            
            results = []
            for proc_name in validation_list:
                try:
                    print(f"Running {proc_name}...")
                    cursor.execute(f"EXEC {proc_name}")
                    single_proc_results = cursor.fetchall()
                    
                    # Get column names from cursor description
                    columns = [column[0] for column in cursor.description] if single_proc_results else []
                    
                    results.append({
                        'procedure_name': proc_name,
                        'columns': columns,
                        'procedure_output': [list(row) for row in single_proc_results] if single_proc_results else [],
                        'message': "Validation failed" if single_proc_results else "Validation passed"
                    })
                    
                except pyodbc.Error as e:
                    print(f"Error running {proc_name}: {str(e)}")
            
            if results:
                for single_result in results:
                    print(f"\n{single_result['procedure_name']}:")
                    if single_result['procedure_output']:
                        print(f"Validation failed - found {len(single_result['procedure_output'])} results:")
                        # Print column headers
                        print(single_result['columns'])
                        # Print each row
                        for row in single_result['procedure_output']:
                            print(row)
                    else:
                        print("Validation passed")
            else:
                print("Validation passed")
        return results
            
    def procedure_lookup(self, 
                        procedures: Union[List[str], str] = 'all', 
                        show_parameters: bool = True) -> None:
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            
            query = """
            SELECT 
                ProcedureName,
                ProcedureDescription,
                Parameter1_Value, Parameter1_Name, Parameter1_Description,
                Parameter2_Value, Parameter2_Name, Parameter2_Description,
                Parameter3_Value, Parameter3_Name, Parameter3_Description,
                Parameter4_Value, Parameter4_Name, Parameter4_Description,
                Parameter5_Value, Parameter5_Name, Parameter5_Description,
                Parameter6_Value, Parameter6_Name, Parameter6_Description,
                Parameter7_Value, Parameter7_Name, Parameter7_Description,
                Parameter8_Value, Parameter8_Name, Parameter8_Description
            FROM ValidationProceduresParameters
            """
            
            if procedures != 'all':
                placeholders = ','.join('?' * len(procedures))
                query += f" WHERE ProcedureName IN ({placeholders})"
                cursor.execute(query, procedures)
            else:
                cursor.execute(query)
            
            for row in cursor.fetchall():
                print(f"\nProcedure: {row[0]}")
                print(f"Description: {row[1]}")
                
                if show_parameters:
                    print("\nParameters:")
                    for i in range(8):  # 8 parameters
                        value_idx = 2 + i * 3
                        if row[value_idx] is not None:  # Only show non-null parameters
                            name = row[value_idx + 1]
                            desc = row[value_idx + 2]
                            print(f"  {name}:")
                            print(f"    Value: {row[value_idx]}")
                            print(f"    Description: {desc}")
                print("-" * 80)

    def procedure_update(self, 
                        procedure_name: str, 
                        parameter_name: str, 
                        new_value: float) -> None:
        with pyodbc.connect(self.conn_str) as conn:
            cursor = conn.cursor()
            
            # Get full row
            cursor.execute("""
                SELECT 
                    Parameter1_Name, Parameter2_Name, Parameter3_Name, 
                    Parameter4_Name, Parameter5_Name, Parameter6_Name,
                    Parameter7_Name, Parameter8_Name
                FROM ValidationProceduresParameters 
                WHERE ProcedureName = ?
            """, procedure_name)
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Procedure {procedure_name} not found")
            
            # Find matching parameter
            for i, name in enumerate(row, 1):
                if name == parameter_name:
                    value_column = f"Parameter{i}_Value"
                    cursor.execute(f"""
                        UPDATE ValidationProceduresParameters 
                        SET {value_column} = ?
                        WHERE ProcedureName = ?
                    """, new_value, procedure_name)
                    conn.commit()
                    print(f"Updated {parameter_name} to {new_value} for {procedure_name}")
                    return
            
            raise ValueError(f"Parameter {parameter_name} not found for {procedure_name}")

if __name__ == "__main__":
    vm = ValidationManager()
    # vm.procedure_lookup()
    # vm.procedure_update('ValidateTradeUpsampled', 'sampling_rate', 60)
    # vm.procedure_lookup()
    vm.run_validation()
