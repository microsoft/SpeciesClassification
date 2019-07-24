var searchString = null,
    nextSearchRange = 0,
    isSearch = !1;

function SearchSubmit() {
    $("#search-submit").click(function() {
        Search(), BlockUI(!0), 
        ProgressLoadingMsgs("Searching images..."), 
        $("#load-more-images").show()
    })
}

function SearchEnterKeyPressed() {
    $("#search-input").keypress(function(e) {
        if (13 == e.which) 
        return $("#search-submit").click(), !1
    })
}

function Search() {
    try {
        searchString = $("#search-input").val(), 
        nextSearchRange = 0, 
        isSearch = !0, 
        BlockUI(!0), $.ajax({
            url: "get_search_results",
            data: {
                searchString: searchString
            }
        }).done(function(e) {
            UnBlockUI(), e && 0 != e.length ? (
             ClearGallery(), 
             CreateImageColumns(), 
             AddGalleryTemplate(e, !1, !1)) : 
             alert("Sorry! Could not find any search result for search term: " + searchString)
        }).fail(function(e, r, a) {
            UnBlockUI(), 
            e.getAllResponseHeaders() 
            && (alert("An error occurred while searching data,\
             please check console for more details"), 
            console.log("An occurred while searching data"), 
            console.log("Error details:"), 
            console.log(a))
        })
    } catch (e) {
        alert("An error occurred while searching data,\
        please check console for more details"), 
        console.log("An error occurred while searching data"), 
        console.log("Error details:"), 
        console.log(e)
    }
}

function GetMoreSearchImages() {
    try {
        nextSearchRange += 4, BlockUI(), $.ajax({
            url: "get_more_search_images",
            data: {
                startRange: nextSearchRange,
                numberOfImages: 8
            }
        }).done(function(e) {
            UnBlockUI(), 0 == e.data ? HideLoadMoreImages() : 0 == e.end_range ? (
             AddGalleryTemplate(data, !1), 
             HideLoadMoreImages()) : 
             (nextSearchRange = e.end_range, AddGalleryTemplate(e.data, !0, !0), 
             ShowLoadMoreImages())
        }).fail(function(e, r, a) {
            UnBlockUI(), 
            e.getAllResponseHeaders() 
            && (alert("An error occurred while retrieving search images,\
            please check console for more details"), 
            console.log("An occurred while retrieving search images"), 
            console.log("Error details:"), 
            console.log(a))
        })
    } catch (e) {
        alert("An error occurred while retrieving search images,\
        please check console for more details"), 
        console.log("An error occurred while retrieving search images"), 
        console.log("Error details:"), 
        console.log(e)
    }
}

function ClearGallery() {
    $("#gallery-items").html("")
}