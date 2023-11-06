class GitHubRepoSearchQueryBuilder:
    def __init__(self):
        self.query_params = {}
        
    def search_in(self, value, name=False, description=False, readme=False, topics=False):
        search_in = ''
        if name:
            search_in += 'in:name,'
        if description:
            if search_in == '':
                search_in += 'in:description,'
            else:
                search_in += 'description,'
        if readme:
            if search_in == '':
                search_in += 'in:readme,'
            else:
                search_in += 'readme,'
        if topics:
            if search_in == '':
                search_in += 'in:topics,'
            else:
                search_in += 'topics'
        if search_in != '':
            search_in = ' ' + search_in
        self.query_params[value] = search_in[:-1]
        return self
    

    def repo(self, owner, name):
        self.query_params['repo:'] = f'{owner}/{name}'
        return self
    
    def user(self, value):
        self.query_params['user:'] = value
        return self

    def org(self, value):
        self.query_params['org:'] = value
        return self

    def size(self, value):
        self.query_params['size:'] = value
        return self

    def size_gt(self, value):
        self.query_params['size:'] = f'>{value}'
        return self
    
    def size_lt(self, value):
        self.query_params['size:'] = f'<{value}'
        return self

    def size_range(self, min_value, max_value):
        self.query_params['size:'] = f'{min_value}..{max_value}'
        return self
    
    def followers_gt(self, value):
        self.query_params['followers:'] = f'>{value}'
        return self
    
    def followers_range(self, min_value, max_value):
        self.query_params['followers:'] = f'{min_value}..{max_value}'
        return self

    def forks(self, value):
        self.query_params['forks:'] = value
        return self

    def forks_gt(self, value):
        self.query_params['forks:'] = f'>{value}'
        return self

    def forks_lt(self, value):
        self.query_params['forks:'] = f'<{value}'
        return self

    def forks_range(self, min_value, max_value):
        self.query_params['forks:'] = f'{min_value}..{max_value}'
        return self

    def stars(self, value):
        self.query_params['stars:'] = value
        return self

    def stars_gt(self, value):
        self.query_params['stars:'] = f'>{value}'
        return self

    def stars_lt(self, value):
        self.query_params['stars:'] = f'<{value}'
        return self

    def stars_range(self, min_value, max_value):
        self.query_params['stars:'] = f'{min_value}..{max_value}'
        return self

    def created(self, value):
        self.query_params['created:'] = value
        return self

    def created_gt(self, value):
        self.query_params['created:'] = f'>{value}'
        return self

    def created_lt(self, value):
        self.query_params['created:'] = f'<{value}'
        return self

    def created_range(self, min_value, max_value):
        self.query_params['created:'] = f'{min_value}..{max_value}'
        return self

    def pushed(self, value):
        self.query_params['pushed:'] = value
        return self

    def pushed_gt(self, value):
        self.query_params['pushed:'] = f'>{value}'
        return self

    def pushed_lt(self, value):
        self.query_params['pushed:'] = f'<{value}'
        return self

    def pushed_range(self, min_value, max_value):
        self.query_params['pushed:'] = f'{min_value}..{max_value}'
        return self

    def language(self, value):
        self.query_params['language:'] = value
        return self

    def topic(self, value):
        self.query_params['topic:'] = value
        return self

    def number_of_topics(self, value):
        self.query_params['topic:'] = f'{value}'
        return self

    def number_of_topics_gt(self, value):
        self.query_params['topic:'] = f'>{value}'
        return self

    def license(self, value):
        self.query_params['license:'] = value
        return self

    def is_public_private(self, value):
        self.query_params['is:'] = value
        return self

    def mirror(self, value):
        self.query_params['mirror:'] = value
        return self

    def template(self, value):
        self.query_params['template:'] = value
        return self

    def archived(self, value):
        self.query_params['archived:'] = value
        return self

    def good_first_issues(self, value):
        self.query_params['good-first-issues:'] = f'>{value}'
        return self

    def help_wanted_issues(self, value):
        self.query_params['help-wanted-issues:'] = f'>{value}'
        return self

    
    def build(self):
        full_query = ''
        for key, value in self.query_params.items():
            full_query += f'{key}{value} '
        return full_query.strip()

    def help(self):
        help_message = '''
        GitHubRepoSearchQueryBuilder
        ----------------------------
        This class helps you to build a GitHub search query.
        You can use this class to build a GitHub search query
        and then use it to search for repositories on GitHub.
        The class has methods to build a query for all the
        search parameters that are available on GitHub.
        The class also has a method to build the full query
        string.
        ----------------------------
        Methods:
        ----------------------------
        search_in(value, name=False, description=False, readme=False, topics=False):
            This method is used to specify the fields in which
            you want to search for the value. You can specify
            multiple fields. The value parameter is the value
            that you want to search for. The name, description,
            readme and topics parameters are used to specify
            the fields in which you want to search for the value.
            The value parameter is required. The other parameters
            are optional. If you don't specify any of the optional
            parameters, the search will be done in all the fields.
            If you specify multiple fields, the search will be
            done in all the fields.

        repo(owner, name):
            This method is used to specify the repository that
            you want to search for. The owner parameter is the
            owner of the repository and the name parameter is
            the name of the repository. Both the parameters are
            required.

        user(value):
            This method is used to specify the user that you
            want to search for. The value parameter is the
            username of the user. The value parameter is required.

        org(value):
            This method is used to specify the organization
            that you want to search for. The value parameter
            is the name of the organization. The value parameter
            is required.

        size(value):
            This method is used to specify the size of the
            repository that you want to search for. The value
            parameter is the size of the repository. The value
            parameter is required.

        size_gt(value):
            This method is used to specify the minimum size
            of the repository that you want to search for. The
            value parameter is the minimum size of the repository.
            The value parameter is required.

        size_lt(value):
            This method is used to specify the maximum size
            of the repository that you want to search for. The
            value parameter is the maximum size of the repository.
            The value parameter is required.

        size_range(min_value, max_value):
            This method is used to specify the range of the
            size of the repository that you want to search for.
            The min_value parameter is the minimum size of the
            repository and the max_value parameter is the maximum
            size of the repository. Both the parameters are required.

        followers_gt(value):
            This method is used to specify the minimum number
            of followers of the user that you want to search for.
            The value parameter is the minimum number of followers
            of the user. The value parameter is required.

        followers_range(min_value, max_value):
            This method is used to specify the range of the
            number of followers of the user that you want to
            search for. The min_value parameter is the minimum
            number of followers of the user and the max_value
            parameter is the maximum number of followers of the
            user. Both the parameters are required.

        forks(value):
            This method is used to specify the number of forks
            of the repository that you want to search for. The
            value parameter is the number of forks of the repository.
            The value parameter is required.

        forks_gt(value):
            This method is used to specify the minimum number
            of forks of the repository that you want to search for.
            The value parameter is the minimum number of forks of
            the repository. The value parameter is required.

        forks_lt(value):
            This method is used to specify the maximum number
            of forks of the repository that you want to search for.
            The value parameter is the maximum number of forks of
            the repository. The value parameter is required.

        forks_range(min_value, max_value):
            This method is used to specify the range of the
            number of forks of the repository that you want to
            search for. The min_value parameter is the minimum
            number of forks of the repository and the max_value
            parameter is the maximum number of forks of the
            repository. Both the parameters are required.

        stars(value):
            This method is used to specify the number of stars
            of the repository that you want to search for. The
            value parameter is the number of stars of the repository.
            The value parameter is required.

        stars_gt(value):
            This method is used to specify the minimum number
            of stars of the repository that you want to search for.
            The value parameter is the minimum number of stars of
            the repository. The value parameter is required.

        stars_lt(value):
            This method is used to specify the maximum number
            of stars of the repository that you want to search for.
            The value parameter is the maximum number of stars of
            the repository. The value parameter is required.

        stars_range(min_value, max_value):
            This method is used to specify the range of the
            number of stars of the repository that you want to
            search for. The min_value parameter is the minimum
            number of stars of the repository and the max_value
            parameter is the maximum number of stars of the
            repository. Both the parameters are required.

        created(value):
            This method is used to specify the date on which
            the repository that you want to search for was created.
            The value parameter is the date on which the repository
            was created. The value parameter is required.

        created_gt(value):
            This method is used to specify the date after which
            the repository that you want to search for was created.
            The value parameter is the date after which the repository
            was created. The value parameter is required.

        created_lt(value):
            This method is used to specify the date before which
            the repository that you want to search for was created.
            The value parameter is the date before which the repository
            was created. The value parameter is required.
            
        created_range(min_value, max_value):
            This method is used to specify the range of the date
            on which the repository that you want to search for
            was created. The min_value parameter is the date before
            which the repository was created and the max_value
            parameter is the date after which the repository was
            created. Both the parameters are required.

        pushed(value):
            This method is used to specify the date on which
            the repository that you want to search for was pushed.
            The value parameter is the date on which the repository
            was pushed. The value parameter is required.

        pushed_gt(value):
            This method is used to specify the date after which
            the repository that you want to search for was pushed.
            The value parameter is the date after which the repository
            was pushed. The value parameter is required.

        pushed_lt(value):
            This method is used to specify the date before which
            the repository that you want to search for was pushed.
            The value parameter is the date before which the repository
            was pushed. The value parameter is required.

        pushed_range(min_value, max_value):
            This method is used to specify the range of the date
            on which the repository that you want to search for
            was pushed. The min_value parameter is the date before
            which the repository was pushed and the max_value
            parameter is the date after which the repository was
            pushed. Both the parameters are required.

        language(value):
            This method is used to specify the language of the
            repository that you want to search for. The value
            parameter is the language of the repository. The
            value parameter is required.

        topic(value):
            This method is used to specify the topic of the
            repository that you want to search for. The value
            parameter is the topic of the repository. The
            value parameter is required.

        number_of_topics(value):
            This method is used to specify the number of topics
            of the repository that you want to search for. The
            value parameter is the number of topics of the repository.
            The value parameter is required.

        number_of_topics_gt(value):
            This method is used to specify the minimum number
            of topics of the repository that you want to search for.
            The value parameter is the minimum number of topics of
            the repository. The value parameter is required.

        license(value):
            This method is used to specify the license of the
            repository that you want to search for. The value
            parameter is the license of the repository. The
            value parameter is required.

        is_public_private(value):
            This method is used to specify whether the repository
            that you want to search for is public or private. The
            value parameter is the type of the repository. The
            value parameter is required.

        mirror(value):
            This method is used to specify whether the repository
            that you want to search for is a mirror. The value
            parameter is the type of the repository. The
            value parameter is required.

        template(value):
            This method is used to specify whether the repository
            that you want to search for is a template. The value
            parameter is the type of the repository. The
            value parameter is required.

        archived(value):
            This method is used to specify whether the repository
            that you want to search for is archived. The value
            parameter is the type of the repository. The
            value parameter is required.

        good_first_issues(value):
            This method is used to specify the minimum number
            of good first issues of the repository that you want
            to search for. The value parameter is the minimum
            number of good first issues of the repository. The
            value parameter is required.

        help_wanted_issues(value):
            This method is used to specify the minimum number
            of help wanted issues of the repository that you want
            to search for. The value parameter is the minimum
            number of help wanted issues of the repository. The
            value parameter is required.

        build():
            This method is used to build the full query string.
            This method returns the full query string.

        help():
            This method is used to print the help message.
        '''
        return help_message