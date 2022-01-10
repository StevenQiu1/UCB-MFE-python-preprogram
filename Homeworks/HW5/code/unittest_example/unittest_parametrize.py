import pytest

seq = [1, 2, 3]


@pytest.fixture(params=seq)
def test_data(request):
    print("参数")
    return request.param


class TestData:
    def test_1(self, test_data):
        print("用例", test_data)


if __name__ == '__main__':
    pytest.main()