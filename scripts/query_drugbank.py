import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('data/drugbank.db')
cursor = conn.cursor()

# Example 1: Simple query to count records
cursor.execute("SELECT COUNT(*) FROM drugbank_drug")
count = cursor.fetchone()[0]
print(f"Total drugs in database: {count}")

# Example 2: Get column names from a table
cursor.execute("PRAGMA table_info(drugbank_drug)")
columns = cursor.fetchall()
print("\nColumns in drugbank_drug table:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Example 3: Query with filtering
print("\nExample drug information:")
cursor.execute("""
    SELECT name, type, description 
    FROM drugbank_drug 
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"Name: {row[0]}")
    print(f"Type: {row[1]}")
    print(f"Description: {row[2][:100]}..." if row[2] and len(row[2]) > 100 else f"Description: {row[2]}")
    print()

# Example 4: Using pandas to query and analyze data
df = pd.read_sql_query("""
    SELECT name, type 
    FROM drugbank_drug 
    LIMIT 10
""", conn)
print("\nPandas DataFrame example:")
print(df)

# Close the connection
conn.close()

print("\nTo run your own custom query, modify this script or use:")
print("sqlite3 data/drugbank.db \"YOUR SQL QUERY HERE\"") 