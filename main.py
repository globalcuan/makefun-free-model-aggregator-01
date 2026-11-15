"""Serve a deterministic free-model aggregation API.
Discover provider catalogs and route compatible requests with wire translation.
Poll the competition environment and stop after its workload is complete."""
import http.server
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request


def load_brief():
    path = os.environ.get('MAKEFUN_PROMPT_FILE')
    if not path:
        return {}, {}, {}
    try:
        raw = open(path, encoding='utf-8').read()
    except OSError:
        return {}, {}, {}
    docs = []
    decoder = json.JSONDecoder()
    for schema in ('free-model-aggregator-01/providers/v1', 'free-model-aggregator-01/ratings/v1', 'free-model-aggregator-01/workload/v1'):
        pos = raw.find(schema)
        if pos < 0:
            docs.append({})
            continue
        start = raw.rfind('{', 0, pos)
        while start >= 0:
            try:
                obj, _ = decoder.raw_decode(raw[start:])
                if obj.get('schema') == schema:
                    docs.append(obj)
                    break
            except (ValueError, AttributeError):
                pass
            start = raw.rfind('{', 0, start)
        else:
            docs.append({})
    return tuple(docs)


def get_json(url, headers=None, timeout=3):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def post_bytes(url, payload, headers, timeout=20):
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get('Content-Type', '')


def auth_headers(p):
    key = p.get('key') or (os.environ.get(p.get('key_env', ''), '') if p.get('key_env') else '')
    auth = p.get('auth', 'none')
    h = {}
    if key and auth == 'bearer': h['Authorization'] = 'Bearer ' + key
    elif key and auth == 'x-api-key': h['x-api-key'] = key
    elif key and auth == 'x-goog-api-key': h['x-goog-api-key'] = key
    if p.get('wire') == 'anthropic': h['anthropic-version'] = '2023-06-01'
    return h


def build_catalog(directory, ratings, origin):
    instances = {}
    try:
        inst = get_json(origin.rstrip('/') + '/instance')
        instances = {p.get('id'): p for p in inst.get('providers', [])}
    except Exception:
        pass
    rows = ratings.get('ratings', [])
    catalog = {'providers': []}
    for p in directory.get('providers', []):
        live = dict(p)
        live.update(instances.get(p.get('id'), {}))
        base = live.get('base_url', '').rstrip('/')
        wire = p.get('wire', '')
        endpoint = base + ('/v1beta/models' if wire == 'gemini' else '/v1/models')
        models = []
        try:
            data = get_json(endpoint, auth_headers(p))
            listed = data.get('models', []) if wire == 'gemini' else data.get('data', [])
            for m in listed:
                mid = m.get('name', '') if wire == 'gemini' else m.get('id', '')
                if wire == 'gemini' and mid.startswith('models/'): mid = mid[7:]
                rr = next((x for x in rows if x.get('provider') == p.get('id') and x.get('model') == mid), None)
                models.append({'id': mid, 'free': bool(rr and rr.get('free')), 'context_length': rr.get('context_length') if rr else None})
        except Exception:
            for rr in rows:
                if rr.get('provider') == p.get('id') and rr.get('free', True):
                    models.append({'id': rr.get('model'), 'free': True, 'context_length': rr.get('context_length')})
        catalog['providers'].append({'id': p.get('id'), 'base_url': p.get('path'), 'key_env': p.get('key_env'), 'wire': wire, 'models': models})
    return catalog, instances


def text_of(content):
    if isinstance(content, str): return content
    if isinstance(content, list): return '\n'.join(str(x.get('text', '')) for x in content if isinstance(x, dict) and x.get('type') == 'text')
    return '' if content is None else str(content)


def openai_to_anthropic(body):
    out = {'model': body.get('model'), 'max_tokens': body.get('max_tokens', 1024)}
    system = []
    msgs = []
    for m in body.get('messages', []):
        role, content = m.get('role'), m.get('content')
        if role == 'system': system.append(text_of(content)); continue
        blocks = []
        if content:
            if isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text': blocks.append({'type':'text','text':part.get('text','')})
                    elif part.get('type') == 'image_url':
                        u = part.get('image_url', {}).get('url', '')
                        mat = re.match(r'data:([^;,]+);base64,(.*)', u, re.S)
                        source = {'type':'base64','media_type':mat.group(1),'data':mat.group(2)} if mat else {'type':'url','url':u}
                        blocks.append({'type':'image','source':source})
            else: blocks.append({'type':'text','text':str(content)})
        for tc in m.get('tool_calls', []):
            fn = tc.get('function', {})
            try: args = json.loads(fn.get('arguments', '{}'))
            except Exception: args = {}
            blocks.append({'type':'tool_use','id':tc.get('id','call_1'),'name':fn.get('name',''),'input':args})
        if role == 'tool':
            msgs.append({'role':'user','content':[{'type':'tool_result','tool_use_id':m.get('tool_call_id',''),'content':text_of(content)}]})
        elif role in ('user','assistant'): msgs.append({'role':role,'content':blocks})
    if system: out['system'] = '\n'.join(system)
    out['messages'] = msgs
    if body.get('tools'):
        out['tools'] = [{'name':t.get('function',{}).get('name'), 'description':t.get('function',{}).get('description',''), 'input_schema':t.get('function',{}).get('parameters',{})} for t in body['tools']]
    tc = body.get('tool_choice')
    if tc == 'auto': out['tool_choice']={'type':'auto'}
    elif tc == 'none': out['tool_choice']={'type':'none'}
    elif isinstance(tc,dict): out['tool_choice']={'type':'tool','name':tc.get('function',{}).get('name','')}
    if body.get('stop'): out['stop_sequences'] = body['stop'] if isinstance(body['stop'],list) else [body['stop']]
    if body.get('stream'): out['stream']=True
    return out


def anthropic_to_openai(body):
    msgs=[]
    if body.get('system'): msgs.append({'role':'system','content':body['system']})
    for m in body.get('messages',[]):
        role=m.get('role'); content=m.get('content',''); blocks=content if isinstance(content,list) else [{'type':'text','text':content}]
        outparts=[]; calls=[]
        for b in blocks:
            typ=b.get('type')
            if typ=='text': outparts.append({'type':'text','text':b.get('text','')})
            elif typ=='image':
                src=b.get('source',{}); url='data:%s;base64,%s'%(src.get('media_type','image/png'),src.get('data','')) if src.get('type')=='base64' else src.get('url','')
                outparts.append({'type':'image_url','image_url':{'url':url}})
            elif typ=='tool_use': calls.append({'id':b.get('id','call_1'),'type':'function','function':{'name':b.get('name',''),'arguments':json.dumps(b.get('input',{}),separators=(',',':'))}})
            elif typ=='tool_result': msgs.append({'role':'tool','tool_call_id':b.get('tool_use_id',''),'content':text_of(b.get('content',''))})
        mm={'role':role,'content': (''.join(p['text'] for p in outparts) if all(p.get('type')=='text' for p in outparts) else outparts)}
        if calls: mm['tool_calls']=calls
        if role in ('user','assistant'): msgs.append(mm)
    out={'model':body.get('model'),'messages':msgs,'max_tokens':body.get('max_tokens',1024)}
    if body.get('tools'): out['tools']=[{'type':'function','function':{'name':t.get('name'),'description':t.get('description',''),'parameters':t.get('input_schema',{})}} for t in body['tools']]
    choice=body.get('tool_choice',{}); typ=choice.get('type') if isinstance(choice,dict) else None
    if typ in ('auto','none'): out['tool_choice']=typ
    elif typ=='tool': out['tool_choice']={'type':'function','function':{'name':choice.get('name')}}
    if body.get('stop_sequences'): out['stop']=body['stop_sequences']
    if body.get('stream'): out['stream']=True
    return out


def to_gemini(body, client):
    if client=='openai':
        messages=body.get('messages',[]); system='\n'.join(text_of(m.get('content')) for m in messages if m.get('role')=='system')
    else:
        messages=body.get('messages',[]); system=text_of(body.get('system',''))
    contents=[]; tools=[]
    for m in messages:
        role=m.get('role'); content=m.get('content',''); parts=[]
        if client=='openai' and role=='system': continue
        if isinstance(content,list):
            for b in content:
                typ=b.get('type')
                if typ=='text': parts.append({'text':b.get('text','')})
                elif typ=='image_url':
                    u=b.get('image_url',{}).get('url',''); mat=re.match(r'data:([^;,]+);base64,(.*)',u,re.S)
                    parts.append({'inline_data':{'mime_type':mat.group(1),'data':mat.group(2)}} if mat else {'file_data':{'file_uri':u}})
                elif typ=='image':
                    s=b.get('source',{}); parts.append({'inline_data':{'mime_type':s.get('media_type','image/png'),'data':s.get('data','')}} if s.get('type')=='base64' else {'file_data':{'file_uri':s.get('url','')}})
                elif typ=='tool_use': parts.append({'functionCall':{'name':b.get('name'),'args':b.get('input',{})}})
                elif typ=='tool_result': parts.append({'functionResponse':{'name':'tool','response':{'result':text_of(b.get('content',''))}}})
        else:
            if content: parts.append({'text':str(content)})
            if role=='tool': parts=[{'functionResponse':{'name':'tool','response':{'result':str(content)}}}]
        for tc in m.get('tool_calls',[]):
            fn=tc.get('function',{}); 
            try: args=json.loads(fn.get('arguments','{}'))
            except Exception: args={}
            parts.append({'functionCall':{'name':fn.get('name'),'args':args}})
        if parts: contents.append({'role':'model' if role=='assistant' else 'user','parts':parts})
    if client=='openai': ts=body.get('tools',[]); mx=body.get('max_tokens',1024); stop=body.get('stop'); jsonmode=body.get('response_format',{}).get('type')=='json_object'
    else: ts=body.get('tools',[]); mx=body.get('max_tokens',1024); stop=body.get('stop_sequences'); jsonmode=False
    for t in ts:
        fn=t.get('function',t); tools.append({'name':fn.get('name'),'description':fn.get('description',''),'parameters':fn.get('parameters',fn.get('input_schema',{}))})
    out={'contents':contents,'generationConfig':{'maxOutputTokens':mx}}
    if stop: out['generationConfig']['stopSequences']=stop if isinstance(stop,list) else [stop]
    if jsonmode: out['generationConfig']['responseMimeType']='application/json'
    if system: out['systemInstruction']={'parts':[{'text':system}]}
    if tools: out['tools']=[{'functionDeclarations':tools}]
    return out


def parse_provider(wire, data):
    text=''; calls=[]; stop=None; usage={'input':0,'output':0}
    if wire=='openai':
        ch=(data.get('choices') or [{}])[0]; msg=ch.get('message',{}); text=msg.get('content') or ''; stop=ch.get('finish_reason')
        for tc in msg.get('tool_calls',[]): calls.append({'id':tc.get('id','call_1'),'name':tc.get('function',{}).get('name',''),'arguments':tc.get('function',{}).get('arguments','{}')})
        u=data.get('usage',{}); usage={'input':u.get('prompt_tokens',0),'output':u.get('completion_tokens',0)}
    elif wire=='anthropic':
        for b in data.get('content',[]):
            if b.get('type')=='text': text+=b.get('text','')
            if b.get('type')=='tool_use': calls.append({'id':b.get('id','call_1'),'name':b.get('name',''),'arguments':json.dumps(b.get('input',{}),separators=(',',':'))})
        stop=data.get('stop_reason'); u=data.get('usage',{}); usage={'input':u.get('input_tokens',0),'output':u.get('output_tokens',0)}
    else:
        c=(data.get('candidates') or [{}])[0]; stop=c.get('finishReason'); u=data.get('usageMetadata',{}); usage={'input':u.get('promptTokenCount',0),'output':u.get('candidatesTokenCount',0)}
        for p in c.get('content',{}).get('parts',[]):
            if 'text' in p: text+=p['text']
            if 'functionCall' in p: calls.append({'id':'call_1','name':p['functionCall'].get('name',''),'arguments':json.dumps(p['functionCall'].get('args',{}),separators=(',',':'))})
    return {'text':text,'tool_calls':calls,'stop':stop,'usage':usage}


def decode_sse(raw):
    events=[]
    for line in raw.decode('utf-8','replace').splitlines():
        if line.startswith('data:'):
            d=line[5:].strip()
            if d and d!='[DONE]':
                try: events.append(json.loads(d))
                except ValueError: pass
    return events


def parse_stream(wire, raw):
    evs=decode_sse(raw)
    if wire=='openai':
        text=''; calls={}; stop=None; usage={'input':0,'output':0}
        for e in evs:
            if e.get('usage'):
                u=e['usage']; usage={'input':u.get('prompt_tokens',0),'output':u.get('completion_tokens',0)}
            for c in e.get('choices',[]):
                d=c.get('delta',{}); text+=d.get('content') or ''; stop=c.get('finish_reason') or stop
                for t in d.get('tool_calls',[]):
                    x=calls.setdefault(t.get('index',0),{'id':t.get('id','call_1'),'name':'','arguments':''})
                    if t.get('id'): x['id']=t['id']
                    f=t.get('function',{}); x['name']+=f.get('name',''); x['arguments']+=f.get('arguments','')
        return {'text':text,'tool_calls':list(calls.values()),'stop':stop,'usage':usage}
    if wire=='anthropic':
        text=''; blocks={}; stop=None; usage={'input':0,'output':0}
        for e in evs:
            typ=e.get('type')
            if typ=='message_start': usage['input']=e.get('message',{}).get('usage',{}).get('input_tokens',0)
            if typ=='content_block_start': blocks[e.get('index',0)]=e.get('content_block',{})
            if typ=='content_block_delta':
                i=e.get('index',0); d=e.get('delta',{}); b=blocks.setdefault(i,{'type':'text','text':''})
                if d.get('type')=='text_delta': b['text']=b.get('text','')+d.get('text','')
                if d.get('type')=='input_json_delta': b['_json']=b.get('_json','')+d.get('partial_json','')
            if typ=='message_delta': stop=e.get('delta',{}).get('stop_reason'); usage['output']=e.get('usage',{}).get('output_tokens',0)
        for b in blocks.values():
            if b.get('type')=='text': text+=b.get('text','')
        calls=[{'id':b.get('id','call_1'),'name':b.get('name',''),'arguments':b.get('_json') or json.dumps(b.get('input',{}),separators=(',',':'))} for b in blocks.values() if b.get('type')=='tool_use']
        return {'text':text,'tool_calls':calls,'stop':stop,'usage':usage}
    data={}
    for e in evs:
        c=(e.get('candidates') or [{}])[0]; data=c
    return parse_provider('gemini',{'candidates':[data],'usageMetadata':evs[-1].get('usageMetadata',{}) if evs else {}})


def normalized_stop(stop, client, has_calls):
    if has_calls: return 'tool_calls' if client=='openai' else 'tool_use'
    if client=='openai':
        if stop in ('end_turn','stop','STOP'): return 'stop'
        if stop in ('max_tokens','MAX_TOKENS'): return 'length'
        if stop in ('tool_use','function_call'): return 'tool_calls'
        if stop in ('refusal','SAFETY'): return 'content_filter'
        return stop or 'stop'
    return {'stop':'end_turn','STOP':'end_turn','length':'max_tokens','MAX_TOKENS':'max_tokens','tool_calls':'tool_use','function_call':'tool_use','content_filter':'refusal','SAFETY':'refusal'}.get(stop,stop or 'end_turn')


def response_json(answer, client, model, n):
    usage=answer['usage']; stop=normalized_stop(answer['stop'],client,bool(answer['tool_calls']))
    if client=='openai':
        msg={'role':'assistant','content':answer['text'] or (None if answer['tool_calls'] else '')}
        if answer['tool_calls']: msg['tool_calls']=[{'id':x['id'],'type':'function','function':{'name':x['name'],'arguments':x['arguments']}} for x in answer['tool_calls']]
        return {'id':'chatcmpl-'+str(n),'object':'chat.completion','created':0,'model':model,'choices':[{'index':0,'message':msg,'finish_reason':stop}],'usage':{'prompt_tokens':usage['input'],'completion_tokens':usage['output'],'total_tokens':usage['input']+usage['output']}}
    content=[]
    if answer['text']: content.append({'type':'text','text':answer['text']})
    for x in answer['tool_calls']:
        try: args=json.loads(x['arguments'])
        except Exception: args={}
        content.append({'type':'tool_use','id':x['id'],'name':x['name'],'input':args})
    return {'id':'msg_'+str(n),'type':'message','role':'assistant','model':model,'content':content,'stop_reason':stop,'stop_sequence':None,'usage':{'input_tokens':usage['input'],'output_tokens':usage['output']}}


def sse_response(answer, client, model, n):
    obj=response_json(answer,client,model,n)
    if client=='openai':
        base={'id':obj['id'],'object':'chat.completion.chunk','created':0,'model':model}; lines=[]
        def emit(delta,reason=None,choices=True):
            x=dict(base); x['choices']=[{'index':0,'delta':delta,'finish_reason':reason}] if choices else []; return 'data: '+json.dumps(x,separators=(',',':'))+'\n\n'
        lines.append(emit({'role':'assistant','content':''}))
        txt=answer['text']
        for i in range(0,len(txt),40): lines.append(emit({'content':txt[i:i+40]}))
        if answer['tool_calls']: lines.append(emit({'tool_calls':[{'index':i,'id':t['id'],'type':'function','function':{'name':t['name'],'arguments':t['arguments']}} for i,t in enumerate(answer['tool_calls'])]}))
        lines.append(emit({},obj['choices'][0]['finish_reason']))
        x=dict(base); x['choices']=[]; x['usage']=obj['usage']; lines.append('data: '+json.dumps(x,separators=(',',':'))+'\n\n'); lines.append('data: [DONE]\n\n')
        return ''.join(lines)
    lines=[]
    def event(name,data): return 'event: '+name+'\ndata: '+json.dumps(data,separators=(',',':'),ensure_ascii=False)+'\n\n'
    lines.append(event('message_start',{'type':'message_start','message':{'id':obj['id'],'type':'message','role':'assistant','model':model,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':answer['usage']['input'],'output_tokens':0}}}))
    blocks=[]
    if answer['text']: blocks.append({'type':'text','text':answer['text']})
    for t in answer['tool_calls']:
        try: inp=json.loads(t['arguments'])
        except Exception: inp={}
        blocks.append({'type':'tool_use','id':t['id'],'name':t['name'],'input':inp,'_args':t['arguments']})
    if not blocks: blocks=[{'type':'text','text':''}]
    for i,b in enumerate(blocks):
        clean={k:v for k,v in b.items() if not k.startswith('_')}; lines.append(event('content_block_start',{'type':'content_block_start','index':i,'content_block':clean if b['type']=='tool_use' else {'type':'text','text':''}}))
        if b['type']=='text':
            for j in range(0,len(b['text']),40): lines.append(event('content_block_delta',{'type':'content_block_delta','index':i,'delta':{'type':'text_delta','text':b['text'][j:j+40]}}))
        else: lines.append(event('content_block_delta',{'type':'content_block_delta','index':i,'delta':{'type':'input_json_delta','partial_json':b['_args']}}))
        lines.append(event('content_block_stop',{'type':'content_block_stop','index':i}))
    lines.append(event('message_delta',{'type':'message_delta','delta':{'stop_reason':obj['stop_reason'],'stop_sequence':None},'usage':{'output_tokens':answer['usage']['output']}})); lines.append(event('message_stop',{'type':'message_stop'}))
    return ''.join(lines)


def run():
    directory, ratings_doc, workload_doc = load_brief()
    origin=os.environ.get('MAKEFUN_ENV_URL','').rstrip('/')
    catalog, instances=build_catalog(directory,ratings_doc,origin) if origin else ({'providers':[]},{})
    catalog_text=json.dumps(catalog,separators=(',',':'),ensure_ascii=False)
    if '--list' in sys.argv:
        print(catalog_text); return
    ratings=ratings_doc.get('ratings',[]); workload=workload_doc.get('manifest',[])
    ratingmap={(x.get('provider'),x.get('model')):x for x in ratings}
    provider_dir={p.get('id'):p for p in directory.get('providers',[])}
    for p in provider_dir.values():
        if p.get('id') in instances: p['_live']=instances[p['id']]
    refused=set(); counter=0; serial=0; lock=threading.Lock(); inflight=0; last_finished=[0.0]; stopping=threading.Event()
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*a): pass
        def sendbody(self,status,body,ctype='application/json'):
            b=body.encode() if isinstance(body,str) else body
            self.send_response(status); self.send_header('Content-Type',ctype); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
            with lock: last_finished[0]=__import__('time').monotonic()
        def do_GET(self):
            if self.path!='/v1/models': self.sendbody(404,b'{}'); return
            ids=[]
            for p in catalog['providers']:
                for m in p['models']:
                    if m['free'] and m['id'] not in ids: ids.append(m['id'])
            if self.headers.get('x-api-key') is not None:
                data=[{'id':x,'type':'model','display_name':x,'created_at':'2025-01-01T00:00:00Z'} for x in ids]
                self.sendbody(200,json.dumps({'data':data,'has_more':False,'first_id':ids[0] if ids else None,'last_id':ids[-1] if ids else None},separators=(',',':')))
            else: self.sendbody(200,json.dumps({'object':'list','data':[{'id':x,'object':'model','created':0,'owned_by':'free'} for x in ids]},separators=(',',':')))
        def do_POST(self):
            nonlocal counter,serial,inflight
            if self.path not in ('/v1/chat/completions','/v1/messages'): self.sendbody(404,b'{}'); return
            client='anthropic' if self.path.endswith('/messages') else 'openai'
            with lock: inflight+=1; counter+=1
            try:
                raw=self.rfile.read(int(self.headers.get('Content-Length','0'))); body=json.loads(raw or b'{}')
                nextid=None
                try: work=get_json(origin+'/work'); nextid=work.get('next')
                except Exception: pass
                entry=next((x for x in workload if x.get('id')==nextid),None)
                if entry is None and workload: entry=workload[(counter-1)%len(workload)]
                reqs=(entry or {}).get('requires',{})
                if not reqs:
                    reqs={'tools':bool(body.get('tools')),'vision':False,'json':body.get('response_format',{}).get('type')=='json_object','min_context':0}
                candidates=[]
                for row in ratings:
                    if not row.get('free'): continue
                    if reqs.get('tools') and not row.get('tools'): continue
                    if reqs.get('vision') and not row.get('vision'): continue
                    if reqs.get('json') and not row.get('json'): continue
                    if row.get('context_length',0)<reqs.get('min_context',0): continue
                    p=provider_dir.get(row.get('provider'))
                    if p: candidates.append((row,p))
                candidates.sort(key=lambda x:x[0].get('rating_milli',0),reverse=True)
                last_error='No eligible free model'
                for allow_refused in (False,True):
                    for row,p in candidates:
                        pid=p.get('id')
                        if not allow_refused and pid in refused: continue
                        wire=p.get('wire'); model=row.get('model'); live=p.get('_live',p); base=live.get('base_url','').rstrip('/')
                        translated=dict(body) if wire==client else (openai_to_anthropic(body) if client=='openai' and wire=='anthropic' else anthropic_to_openai(body) if client=='anthropic' and wire=='openai' else to_gemini(body,client))
                        translated['model']=model
                        streaming=bool(body.get('stream'))
                        headers=auth_headers(p); headers['Content-Type']='application/json'; headers['Accept']='text/event-stream' if streaming else 'application/json'
                        if wire=='anthropic' and self.headers.get('anthropic-version'): headers['anthropic-version']=self.headers['anthropic-version']
                        if wire=='openai': url=base+'/v1/chat/completions'
                        elif wire=='anthropic': url=base+'/v1/messages'
                        else:
                            url=base+'/v1beta/models/'+urllib.parse.quote(model,safe='')+(':streamGenerateContent?alt=sse' if streaming else ':generateContent')
                        try:
                            status,resp,ct=post_bytes(url,json.dumps(translated,separators=(',',':'),ensure_ascii=False).encode(),headers)
                            if status<200 or status>=300: raise RuntimeError('provider status '+str(status)+': '+resp.decode('utf8','replace'))
                            data=parse_stream(wire,resp) if streaming else parse_provider(wire,json.loads(resp.decode('utf8')))
                            serial+=1; answer=response_json(data,client,model,serial)
                            if streaming: self.sendbody(200,sse_response(data,client,model,serial),'text/event-stream')
                            else: self.sendbody(200,json.dumps(answer,separators=(',',':'),ensure_ascii=False))
                            return
                        except Exception as e:
                            last_error=str(e); refused.add(pid)
                    if candidates and len(refused)>=len(set(x[1].get('id') for x in candidates)): break
                serial+=1
                er={'message':last_error,'type':'api_error'}
                out={'error':er} if client=='openai' else {'type':'error','error':er}
                self.sendbody(502,json.dumps(out,separators=(',',':')))
            finally:
                with lock: inflight-=1; last_finished[0]=__import__('time').monotonic()
    server=http.server.ThreadingHTTPServer(('0.0.0.0',0),Handler); port=server.server_address[1]
    try:
        req=urllib.request.Request(origin+'/register',data=json.dumps({'protocol_version':1,'port':port,'catalog':catalog},separators=(',',':'),ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method='POST')
        urllib.request.urlopen(req,timeout=5).read()
    except Exception: pass
    def poll():
        while True:
            try:
                if get_json(origin+'/work').get('done'):
                    with lock: ready=inflight==0 and __import__('time').monotonic()-last_finished[0]>=0.5
                    if ready:
                        stopping.set(); os._exit(0)
            except Exception: pass
            stopping.wait(0.25)
    threading.Thread(target=poll,daemon=True).start()
    server.serve_forever(poll_interval=0.1)


if __name__=='__main__':
    run()
