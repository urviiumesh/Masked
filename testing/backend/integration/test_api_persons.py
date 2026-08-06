def test_list_persons(client):
    res = client.get("/api/persons")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    if body:
        person = body[0]
        assert "name" in person
        assert "image_count" in person
        assert "embedding_count" in person
        assert "is_unknown" in person


def test_list_persons_search(client):
    all_res = client.get("/api/persons")
    assert all_res.status_code == 200
    people = all_res.json()
    if not people:
        return
    name = people[0]["name"]
    res = client.get(f"/api/persons?search={name[:2]}")
    assert res.status_code == 200
    filtered = res.json()
    assert all(name[:2].lower() in p["name"].lower() for p in filtered)


def test_get_person_not_found(client):
    res = client.get("/api/persons/DefinitelyMissingPersonXYZ")
    assert res.status_code == 404


def test_get_person_existing(client):
    people = client.get("/api/persons").json()
    if not people:
        return
    name = people[0]["name"]
    res = client.get(f"/api/persons/{name}")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == name
    assert "images" in body


def test_delete_person_not_found(client):
    res = client.delete("/api/persons/DefinitelyMissingPersonXYZ")
    assert res.status_code == 404
