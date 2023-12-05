class GitHubRepoSearchQueryBuilder:
    def __init__(self):
        self.query_params = {}

    def init_from_args(self, args):
        value = args['value']

        search_in = ''
        prefix = ' in:'
        if args['name']:
            search_in += f'{prefix}name'
            prefix = ','
        if args['description']:
            search_in += f'{prefix}description'
            prefix = ','
        if args['readme']:
            search_in += f'{prefix}readme'
            prefix = ','
        if args['topics']:
            search_in += f'{prefix}topics'
            prefix = ','

        self.query_params[value] = search_in

        if 'owner' in args and 'name' in args:
            self.query_params['repo:'] = f'{args["owner"]}/{args["name"]}'
            
        if 'user' in args:
            self.query_params['user:'] = args['user']

        if 'min_size' in args and 'max_size' in args:
            self.query_params['size:'] = f'{args["min_size"]}..{args["max_size"]}'
        elif 'min_size' in args:
            self.query_params['size:'] = f'>{args["min_size"]}'
        elif 'max_size' in args:
            self.query_params['size:'] = f'<{args["max_size"]}'

        if 'min_forks' in args and 'max_forks' in args:
            self.query_params['forks:'] = f'{args["min_forks"]}..{args["max_forks"]}'
        elif 'min_forks' in args:
            self.query_params['forks:'] = f'>{args["min_forks"]}'
        elif 'max_forks' in args:
            self.query_params['forks:'] = f'<{args["max_forks"]}'

        if 'min_stars' in args and 'max_stars' in args:
            self.query_params['stars:'] = f'{args["min_stars"]}..{args["max_stars"]}'
        elif 'min_stars' in args:
            self.query_params['stars:'] = f'>{args["min_stars"]}'
        elif 'max_stars' in args:
            self.query_params['stars:'] = f'<{args["max_stars"]}'

        if 'min_created' in args and 'max_created' in args:
            self.query_params['created:'] = f'{args["min_created"]}..{args["max_created"]}'
        elif 'min_created' in args:
            self.query_params['created:'] = f'>{args["min_created"]}'
        elif 'max_created' in args:
            self.query_params['created:'] = f'<{args["max_created"]}'

        if 'license' in args:
            self.query_params['license:'] = args['license']

        if 'language' in args:
            self.query_params['language:'] = args['language']

        if 'topic' in args:
            self.query_params['topic:'] = args['topic']

    
    def build(self):
        full_query = ''
        for key, value in self.query_params.items():
            full_query += f'{key}{value} '
        return full_query.strip()