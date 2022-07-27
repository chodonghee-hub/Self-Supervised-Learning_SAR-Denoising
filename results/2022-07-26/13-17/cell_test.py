import pandas as pd
r'''
df = pd.read_excel('./train_log.xlsx', engine = 'openpyxl', sheet_name = 'epochs=100', header=10, usecols = 'C, D, E, F)
print(df)

'''

df1 = pd.read_excel('./train_log.xlsx', sheet_name = 'epochs=100')
df2 = df1[['DATE1', 'DATE2', 'DATE3']]

if not os.path.exists('./test_log.xlsx') :
    with pd.ExccelWriter('./test_log.xlsx', mode = 'w', engine = 'openpyxl') as writer :
        df1.to_excel(writer, index = False, sheet_name = 'epochs=100')
        df2.to_excel(writer, index = False, shhet_name = 'epochs=100', startcol = 12, startrow = 2)

else :
    with pd.ExcelWriter('./test_log.xlsx', mode = 'a', engine = 'openpyxl') as writer :
        df1.to_excel(writer, index = False, sheet_name = 'epochs=100')
        df2.to_excel(writer, index = False, sheet_name = 'epochs=100', startcol = 12, startrow = 2)
    
