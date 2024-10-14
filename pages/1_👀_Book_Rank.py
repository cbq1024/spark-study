import re
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib2 import Path
from pyspark import SparkConf
from pyspark.sql import SparkSession

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention

book_fields_with_first_authors = "book_id,title,SUBSTRING_INDEX(authors, '/', 1) AS first_author,average_rating,language_code,text_reviews_count,publication_date,publisher"
DEBUG = False

# 定义英文列名到中文列名的映射字典
column_name_mapping = {
    "book_id": "书籍编号",
    "title": "书名",
    "first_author": "第一作者",
    "average_rating": "平均评分",
    "language_code": "语言",
    "text_reviews_count": "书面评论数",
    "publication_date": "出版日期",
    "publisher": "出版社"
}


def date_covert(data):
    try:
        converted_date = pd.to_datetime(data, format="%m/%d/%Y")
        return converted_date.strftime("%Y-%m-%d")
    except ValueError:
        return np.nan


def convert_column_names(column_name):
    """
    将驼峰命名法的列名转换为下划线命名法
     e.g. BookID --> book_id
    """
    return re.sub(r'([a-z])([A-Z])', r'\1_\2', column_name).lower()


def process_data(input_dir, output_dir):
    df = pd.read_csv(input_dir, on_bad_lines='skip')

    df.columns = df.columns.str.strip().map(convert_column_names)
    df.columns = df.columns.str.strip()
    df['publication_date'] = df['publication_date'].apply(lambda x: date_covert(x))
    df = df.dropna().drop_duplicates(keep='first')

    # 应用列名映射到中文
    df.rename(columns=column_name_mapping, inplace=True)

    output_file = output_dir / f"{input_dir.stem}.csv"
    print(df['语言'].unique())
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Saved {output_file}")


def average_rating(spark, count):
    """
     根据 average_rating(该书收到的书面文本评论总数) 统计前 count 本书籍
    :param count:
    :param spark: spark instance
    :return: saved file
    """
    book_list = spark.sql(f"SELECT {book_fields_with_first_authors} FROM books \
                                  ORDER BY average_rating DESC")
    return book_list.limit(count)


def num_pages(spark, count):
    """
     根据 num_pages(主书页数) 统计前 count 本书籍
    :param spark: spark instance
    :param count: top sum
    :return: saved file
    """
    book_list = spark.sql(f"SELECT {book_fields_with_first_authors} FROM books \
                              ORDER BY num_pages DESC")
    return book_list.limit(count)


def ratings_count(spark, count):
    """
     根据 ratings_count(收到的唯一评分数量) 统计前 count 本书籍
    :param spark: spark instance
    :param count: top sum
    :return: saved file
    """
    book_list = spark.sql(f"SELECT {book_fields_with_first_authors} FROM books \
                              ORDER BY ratings_count DESC")
    return book_list.limit(count)


def text_reviews_count(spark, count):
    """
     根据 text_reviews_count(该书收到的书面文本评论总数) 统计前 count 本书籍
    :param spark: spark instance
    :param count: top sum
    :return: saved file
    """
    book_list = spark.sql(f"SELECT {book_fields_with_first_authors} FROM books \
                          ORDER BY text_reviews_count DESC")
    return book_list.limit(count)


@st.cache_resource
def get_spark_connection(cleaned_data_path):
    progress_bar = st.progress(0)  # 创建进度条
    # 模拟进度逐步更新
    for i in range(1, 101, 20):
        time.sleep(0.1)  # 每次睡眠模拟任务耗时
        progress_bar.progress(i)  # 更新进度条
    spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
    books_df = spark.read.csv(str(cleaned_data_path / "books-kaggle-mohamadreza-momeni.csv"), header=True,
                              inferSchema=True)
    books_df = books_df.repartition(1)
    books_df.createOrReplaceTempView("books")
    progress_bar.progress(100)
    return spark


def main():
    st.title("📚 图书数据分析展示")
    colored_header("🔍 数据处理和展示平台", "通过多个维度展示图书数据", color_name="blue-70")

    st.markdown("> 📦 正在加载资源...")
    add_vertical_space(1)  # 在界面增加一些垂直间距

    # data etl
    root_path = Path(__file__).resolve().parents[1]
    uncleaned_data_path = root_path / "data" / "uncleaned"
    cleaned_data_path = root_path / "data" / "cleaned"

    if not cleaned_data_path.exists():
        for file in uncleaned_data_path.glob("*.csv"):
            process_data(file, cleaned_data_path)

    spark = get_spark_connection(cleaned_data_path)

    if not DEBUG:

        # 添加选择框用于选择统计类型
        analysis_type = st.sidebar.selectbox(
            "📊 选择分析的类别",  # 添加提示和图标
            ("平均评分最高的书籍", "页面最多的书籍", "评分数量最多的书籍", "书面评论最多的书籍"),
            help="选择您要分析的书籍数据类型"
        )

        count = st.sidebar.slider("📈 选择前几本书籍", 5, 5000, 10, help="选择要展示的书籍数量")

        # 按照选择的类型展示结果
        if analysis_type == "平均评分最高的书籍":
            st.write(f"📈 前 {count} 本平均评分最高的书籍")
            result = average_rating(spark, count)
        elif analysis_type == "页面最多的书籍":
            st.write(f"📚 前 {count} 本页面最多的书籍")
            result = num_pages(spark, count)
        elif analysis_type == "评分数量最多的书籍":
            st.write(f"🌟 前 {count} 本评分数量最多的书籍")
            result = ratings_count(spark, count)
        else:
            st.write(f"✍️ 前 {count} 本书面评论最多的书籍")
            result = text_reviews_count(spark, count)

        df = result.toPandas()

        # 应用列名映射到中文
        df.rename(columns=column_name_mapping, inplace=True)

        # 分页展示数据
        items_per_page = 10
        total_pages = len(df) // items_per_page + (1 if len(df) % items_per_page != 0 else 0)
        current_page = st.sidebar.number_input("📄 选择页码", min_value=1, max_value=total_pages, value=1, step=1)

        # 计算当前页的起始和结束索引
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        df_page = df.iloc[start_idx:end_idx]

        # 展示当前页的数据
        st.markdown(f"### 📋 第 {current_page}/{total_pages} 页")
        st.table(df_page)

        # 数据可视化 - 使用多种图表类型展示数据
        # 数据可视化 - 使用多种图表类型展示数据
        st.markdown("### 数据可视化 📊")
        with st.expander("🔍 展开以查看数据可视化图表"):

            # 1. 核密度估计图展示评分的整体分布
            st.markdown("#### 评分的整体分布情况 📊")
            kde_chart = alt.Chart(df).transform_density(
                '平均评分',
                as_=['平均评分', 'density'],
            ).mark_area().encode(
                x='平均评分:Q',
                y='density:Q'
            ).properties(
                width=600,
                height=300,
                title='评分分布密度图'
            )
            st.altair_chart(kde_chart)

            # 2. 累积分布函数 (CDF) 图表展示评分的累积情况
            st.markdown("#### 评分的累积分布情况 (CDF) 📊")
            cdf_chart = alt.Chart(df).transform_window(
                cumulative_count='count()',  # 累加记录数量
                sort=[{'field': '平均评分'}]  # 按“平均评分”排序
            ).mark_line().encode(
                x='平均评分:Q',
                y='cumulative_count:Q'
            ).properties(
                width=600,
                height=300,
                title='评分累积分布函数 (CDF)'
            )
            st.altair_chart(cdf_chart)

            # 3. 评分区间重新划分（增加更多区间划分）
            st.markdown("#### 评分区间重新划分 📊")
            bins = [0, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5]
            labels = ['<2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.25', '4.25-4.5', '4.5-4.75', '4.75-5']
            df['评分区间'] = pd.cut(df['平均评分'], bins=bins, labels=labels)
            binned_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('评分区间:N', title='评分区间'),
                y=alt.Y('count():Q', title='书籍数量'),
                color='评分区间:N',
                tooltip=['评分区间', 'count()']
            ).properties(
                width=600,
                height=300,
                title='重新分组的评分区间分布'
            )
            st.altair_chart(binned_chart)

            # 4. 使用箱线图展示评分分布的细节，包括中位数和四分位数
            st.markdown("#### 评分分布的箱线图 📦")
            box_chart = alt.Chart(df).mark_boxplot().encode(
                y=alt.Y('平均评分:Q', title='评分'),
                color=alt.value('teal')
            ).properties(
                width=600,
                height=300,
                title='评分的箱线图'
            )
            st.altair_chart(box_chart)

            # 5. 评分偏离平均值的情况
            st.markdown("#### 评分偏离平均值的情况 🔍")
            overall_mean_rating = df['平均评分'].mean()
            df['评分偏离'] = df['平均评分'] - overall_mean_rating
            deviation_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('评分偏离:Q', title='评分偏离'),
                y=alt.Y('count():Q', title='书籍数量'),
                color=alt.condition(
                    alt.datum['评分偏离'] > 0,  # 条件：偏离值大于0
                    alt.value('steelblue'),  # 正偏离为蓝色
                    alt.value('orange')  # 负偏离为橙色
                ),
                tooltip=['书名', '评分偏离']
            ).properties(
                width=600,
                height=300,
                title='评分偏离平均值的分布'
            )
            st.altair_chart(deviation_chart)

            # 6. 语言分布条形图
            st.markdown("#### 语言分布情况 🌍")
            language_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('语言:N', title='语言'),
                y=alt.Y('count():Q', title='书籍数量'),
                color='语言:N',
                tooltip=['语言', 'count()']
            ).properties(
                width=600,
                height=300,
                title='不同语言书籍的分布'
            )
            st.altair_chart(language_chart)

            # 7. 评分与书面评论数双轴图
            st.markdown("#### 评分与书面评论数之间的关系 🔄")
            dual_axis_chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('书面评论数:Q', title='书面评论数'),
                y=alt.Y('平均评分:Q', title='平均评分'),
                color='语言:N',
                tooltip=['书名', '平均评分', '书面评论数']
            ).properties(
                width=600,
                height=400,
                title='书面评论数与平均评分的关系'
            )
            st.altair_chart(dual_axis_chart)

        # 增加书籍详细信息展示
        with st.expander("📖 展开以查看书籍详细信息"):
            selected_book = st.selectbox("", df["书名"], help="选择要查看详细信息的书籍")

            # 根据选择的书名展示详细信息
            book_details = df[df["书名"] == selected_book].iloc[0]

            # 使用 Markdown 展示书籍详细信息
            st.markdown(f"""
            ### 📖 {book_details["书名"]}
            - **✍️ 作者**: {book_details["第一作者"]}
            - **⭐ 平均评分**: {book_details["平均评分"]}
            - **🌍 语言**: {book_details["语言"]}
            - **💬 书面评论数**: {book_details["书面评论数"]}
            - **📅 出版日期**: {book_details["出版日期"]}
            - **🏢 出版社**: {book_details["出版社"]}
            """)

    mention(label="🔗 数据源", url="https://www.kaggle.com/mohamadrezamomeni/goodreads-book-datasets-10m", icon="📁")


if __name__ == "__main__":
    main()