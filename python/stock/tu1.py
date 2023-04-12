
import tushare as ts

# 设置tushare的token
ts.set_token("5b4a221d3dd0a8435d322135f49d0db11a2767b8c31a88cd6c46ca49")

# 创建一个pro接口
pro = ts.pro_api()

# 获取上证指数的历史数据

data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

print(data)

df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
print(df)

data2 = pro.ggt_top10(trade_date='20230103')
print(data2)
