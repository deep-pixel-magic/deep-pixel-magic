import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

print('tensorflow version:', tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
logical_devices = tf.config.list_logical_devices('GPU')

print('detected devices:')
for device in physical_devices:
    details = tf.config.experimental.get_device_details(device)
    model = details.get('device_name', 'Unknown')

    print(f'  {model} ({device.name})')

selected_devices = physical_devices[1:3]

print('selected devices:')
for device in selected_devices:
    details = tf.config.experimental.get_device_details(device)
    device_model = details.get('device_name', 'Unknown')

    print('  ', device_model)

selected_logical_devices = logical_devices[1:3]

print ('using logical devices:')
for device in selected_logical_devices:
    print('  ', device)


strategy = tf.distribute.MirroredStrategy(selected_logical_devices)

result = 0

with strategy.scope():
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    result = tf.matmul(a, b)

tf.print(result)
