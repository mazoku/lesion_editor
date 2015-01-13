
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('config.ini')

print config.items('General parameters')
print config.items('Smoothing parameters')
print config.items('Color model parameters')
print config.items('Localization parameters')
