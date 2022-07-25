
# Import Adafruit IO MQTT client.
from Adafruit_IO import MQTTClient

class MQTTSubscriber:
    def __init__(self, aio_key: str, aio_username: str, feed_ids=[]):
        self.aio_key = aio_key
        self.aio_username = aio_username
        self.feed_ids = feed_ids
        
        self.on_connect = None
        self.on_subscribe = None
        self.on_disconnect = None
        self.on_message = None
        
        self.is_connected = False
        
        self.client = None
        
        self.message_predicates = {}
    
    def _connected(self, client):
        print('Connected to Adafruit IO!  Listening for feed changes...')
        is_connected = True
        
        for feed_id in self.feed_ids:
            client.subscribe(feed_id)
            print('Subcribed to feed {}'.format(feed_id))
        
        if self.on_connect:
            self.on_connect(client)

    def _subscribe(self, client, userdata, mid, granted_qos):
        # This method is called when the client subscribes to a new feed.
        print('Subscribed | User:{} Mid:{} QoS:{}'.format(userdata, mid, granted_qos))
        
        if self.on_subscribe:
            self.on_subscribe(client, userdata, mid, granted_qos)

    def _disconnected(self, client):
        # Disconnected function will be called when the client disconnects.
        print('Disconnected from Adafruit IO!')
        is_connected = False
        
        if self.on_disconnect:
            self.on_disconnect(client)

    def _message(self, client, feed_id, payload):
        print('Feed {} received new value: {}'.format(feed_id, payload))
        
        if self.on_message:
            self.on_message(client, feed_id, payload)
        
        if feed_id in self.message_predicates:
            self.message_predicates[feed_id](payload)
    
    def event(self, name: str):
        def decorator(func):
            setattr(self, 'on_{}'.format(name), func)
        return decorator
    
    def message(self, feed_key:str):
        def decorator(func):
            self.message_predicates[feed_key] = func
        return decorator
    
    def start(self):
        # Create an MQTT client instance.
        self.client = MQTTClient(self.aio_username, self.aio_key)

        # Setup the callback functions defined above.
        self.client.on_connect    = self._connected
        self.client.on_disconnect = self._disconnected
        self.client.on_message    = self._message
        self.client.on_subscribe  = self._subscribe

        # Connect to the Adafruit IO server.
        self.client.connect()

        self.client.loop_background()
    
    def stop(self):
        self.client.disconnect()
    
    def __del__(self):
        self.stop()

