import cianparser


def parse_cities(city_name_list: list):

    for city_name in city_name_list:
        print(f'parsing city: {city_name}')
        city_parser = cianparser.CianParser(location=city_name)
        for i in range(1):
            data = city_parser.get_flats(
                deal_type='sale',
                rooms=(i+1),
                with_saving_csv=True,
                additional_settings={'end_page': 2}
            )
            print(len(data))
            print(data[0])


if __name__ == '__main__':
    city_name_list = ['Саратов']
    parse_cities(city_name_list)
