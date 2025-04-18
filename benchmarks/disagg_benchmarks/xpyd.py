import asyncio
import itertools
import os
import time
import uuid

import aiohttp
from aiohttp import web
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60 * 60 * 60)
# conn = aiohttp.TCPConnector(limit=30)
app = Quart(__name__)


async def forward_request(url, data,request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

async def ping(url:str,instance:str):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
        try:
        
            request_id = (
                    instance
                )
            # print('+++---'*12,url,instance)
            async for _ in forward_request(url,None,request_id):
                    continue
            print('+++++++++++++=')

        except Exception as e:
            import sys
            import traceback
            exc_info = sys.exc_info()
            print("Error occurred in disagg prefill proxy server")
            print(e)
            print("".join(traceback.format_exception(*exc_info)))




class RoundRobinProxy:

    def __init__(self, prefill_ports,decode_ports):
        self.prefill_ports = prefill_ports
        self.decode_ports = decode_ports
        self.prefill_port_cycle = itertools.cycle(self.prefill_ports)
        self.decode_port_cycle = itertools.cycle(self.decode_ports)


    async def handle_request(self, request):
        prefill_port = next(self.prefill_port_cycle)
        decode_port = next(self.decode_port_cycle)
        prefill_url = f'http://localhost:{prefill_port}/v1/completions'
        decode_url = f'http://localhost:{decode_port}/v1/completions'
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as session:
            try:
                # print(prefill_url,decode_url)
                # Forward the request
                original_request_data = await request.get_json()
                
                kv_match = [int(prefill_port/100-81),int(decode_port/100-81)]
                # original_request_data['kv_match'] = kv_match
                prefill_request = original_request_data.copy()
                
                prefill_request['max_tokens'] = 1

                request_id = (
                        f"___prefill_addr_{kv_match[0]}___decode_addr_{kv_match[1]}_{random_uuid()}"
                    )

                async for _ in forward_request(prefill_url,
                                       prefill_request,request_id+'_p'):
                    continue
                
                # return decode
                generator = forward_request(decode_url,
                                            original_request_data,request_id+'_d')
                response = await make_response(generator)
                response.timeout = None
                return response

            except Exception as e:
                import sys
                import traceback
                exc_info = sys.exc_info()
                print("Error occurred in disagg prefill proxy server")
                print(e)
                print("".join(traceback.format_exception(*exc_info)))


# proxy = RoundRobinProxy([8100,8200],
#                         [8300]) # 8200 8300

# proxy = RoundRobinProxy([8100],
#                         [8200,8300])

proxy = RoundRobinProxy([8100],
                        [8200])
# asyncio.run(ping("http://localhost:8200/v1/completions","switch_decode"))
# time.sleep(1)

@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    return await proxy.handle_request(request)



if __name__ == '__main__':
    app.run(port=8000)

    