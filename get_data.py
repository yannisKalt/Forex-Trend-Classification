from alpha_vantage.foreignexchange import ForeignExchange
fx = ForeignExchange(key = open('alpha_key.txt').read().rstrip(),
                     output_format = 'pandas')

major_pairs = [('EUR', 'USD'), ('USD', 'JPY'), ('GBP', 'USD'), ('USD', 'CHF')]
commodity_pairs = [('AUD','USD'), ('NZD', 'USD'), ('USD','CAD')]

# NZD/JPY excluded from minor pairs due to data shortage #
minor_pairs = [('EUR', 'GBP'), ('EUR', 'AUD'), ('GBP', 'JPY'),
               ('CHF', 'JPY'), ('GBP', 'CAD')]


pairs = major_pairs + commodity_pairs + minor_pairs
for base, quote in pairs:
   data, meta = fx.get_currency_exchange_daily(from_symbol = base, to_symbol = quote,
                                               outputsize = 'full')
   data.columns = [base + '/' + quote + '_' + s for s in 'OHLC']
   data.to_csv(base + quote + '.csv')                                             
