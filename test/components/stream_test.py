from tensorwatch.stream import Stream


s1 = Stream(stream_name='s1', console_debug=True)
s2 = Stream(stream_name='s2', console_debug=True)
s3 = Stream(stream_name='s3', console_debug=True)

s1.subscribe(s2)
s2.subscribe(s3)

s3.write('S3 wrote this')
s2.write('S2 wrote this')
s1.write('S1 wrote this')

