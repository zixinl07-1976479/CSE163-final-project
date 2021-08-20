"""
Claire Li, CSE 163 AA
Yuhao Zhuang, CSE 163 AC
This program implements the functions for our final proejct,
The Tops of of the Olympics, to solve for the seven questions, including
finding the top teams and top athlete, showing the number of change,
and showing the distribution in plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
sns.set()


def top_10_teams_golds(olympics):
    """
    Takes in the Olympic data, find the top 10 teams with the most
    of the gold medals in the Olympics after WWII. Show the teams and the
    corresponding total medals in a bar chart.
    """
    olympics = olympics[olympics['Medal'] == 'Gold']
    olympics = olympics.drop_duplicates(subset=['Team', 'Year',
                                                'Event', 'Medal'])
    teams = olympics.groupby('Team')['Medal'].count()
    # convert from series to dataframe
    teams = pd.DataFrame({'Team': teams.index, 'Gold': teams.values})
    result = teams.sort_values('Gold', ascending=False).head(10)
    # Draw plot
    sns.barplot(x='Team', y='Gold', color='b', data=result)
    plt.xticks(rotation=-45)
    plt.title('Top 10 Teams with Gold Medals in Olympics after WWII')
    plt.xlabel('Team')
    plt.ylabel('Gold Medals')
    plt.savefig('top_10_teams_golds.png', bbox_inches='tight')


def top_5_summer_gold(olympics):
    """
    Takes in the Olympic data, find the top 5 teams with most of
    gold medals in the summer Olympics from 2000 t0 2016. Plot a line
    chart to show the trend of gold medals changing.
    """
    # Filter season, year, medal, and duplicate events.
    season = olympics['Season'] == 'Summer'
    medal = olympics['Medal'] == 'Gold'
    time = olympics['Year'] >= 2000
    olympics = olympics[season & medal & time]
    olympics = olympics.drop_duplicates(subset=['Team', 'Year',
                                                'Event', 'Medal'])
    # Filter top 5 teams in the recent 5 summer Olympics
    teams = olympics.groupby('Team')['Medal'].count().reset_index()
    teams = teams.rename(columns={'Medal': 'Gold'})
    top_5 = teams.sort_values('Gold', ascending=False).head(5)
    top_5_teams = top_5['Team'].tolist()
    # Find out top 5 teams corresponds to each olympics' total gold medal
    olympics = olympics[olympics.Team.isin(top_5_teams)]
    teams = olympics.groupby(['Team', 'Year'])['Medal'].count().reset_index()
    teams = teams.rename(columns={'Medal': 'Gold'})
    # Draw plot
    sns.relplot(data=teams, x="Year", y="Gold", hue="Team",
                kind="line")
    plt.xticks(np.arange(2000, 2017, step=4))
    plt.title('Top 5 Teams with Gold Medals in Recent 5 Olympics')
    plt.xlabel('Year')
    plt.ylabel('Gold Medals')
    plt.savefig('top_5_summer_golds.png', bbox_inches='tight')


def events(olympics):
    """
    Takes in the Olympic data, calculate the number of events for the
    Olympic from 1984 till 2016. Plot two line charts to show the trend of
    events quantity changing in summer Olympics and winter Olympics.
    """
    # Filter year
    time = olympics['Year'] >= 1984
    olympics = olympics[time]
    # Drop out cols and duplicates
    olympics = olympics[['Year', 'Event', 'Season']]
    olympics = olympics.drop_duplicates(subset=['Year', 'Event', 'Season'])
    # Calculate events in every year
    events = olympics.groupby(['Year',
                               'Season'])['Event'].count().reset_index()
    # setup figures
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    # plot 1
    summer = events[events['Season'] == 'Summer']
    summer.plot(ax=ax1, x='Year', y='Event')
    ax1.set_title('Summer Olympics Events Number')
    # plot 2
    winters = events[events['Season'] == 'Winter']
    winters.plot(ax=ax2, x='Year', y='Event')
    ax2.set_title('Winter Olympics Events Number')
    fig.tight_layout(pad=2.0)
    plt.savefig('events_change.png')


def most_medal_athlete(olympics):
    """
    Takes in the Olympic data and finds the athlete who got most medals.
    """
    # Filter Medal
    olympics = olympics[olympics.Medal.notnull()]
    # Find the athletes
    events = olympics.groupby('Name')['Medal'].count().reset_index()
    most_medal_athlete = events.sort_values('Medal', ascending=False).head(1)
    name = most_medal_athlete['Name'].values[0]
    result = olympics[olympics['Name'] == name]
    print(result)
    return result


def us_medals_by_sports(olympics, sport):
    """
    Takes in the Olympic data and Sports list, show the medals
    distribution in the US by a pie chart.
    """
    us_medals = olympics[olympics['Team'] == 'United States']
    us_medals = us_medals.drop_duplicates(subset=['Event', 'Medal'])
    # Merge two datasets
    merged = sport.merge(us_medals, left_on='Event', right_on='Event',
                         how='left')
    merged = merged.groupby(['Sport'], as_index=False).count()
    label = merged['Sport']
    value = merged['Medal']
    # Plot
    fig = go.Figure(data=[go.Pie(labels=label, values=value,
                    title='Medals Distribution in US')])
    # Normalize figure
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.write_image('us_medals_by_sports.png')
    # fig.show()


def su_athletes_age_sex(olympics):
    """
    Takes in the Olympic data, show the average age of athletes from 1976 to
    2016. Plot 3 graphs separately with 'all atheletes average age',
    'female athletes average age', and 'male athletes average age' when
    clicking the button.
    """
    olympics = olympics[olympics['Year'] >= 1976]
    su_olympics = olympics[olympics['Season'] == 'Summer']
    athletes_age_sex = su_olympics.drop_duplicates(subset=['Age',
                                                           'Sex', 'Year'])
    ave_sex_year = athletes_age_sex.groupby(['Year', 'Sex'],
                                            as_index=False)['Age'].mean()
    # Plot
    fig = px.line(ave_sex_year, x="Year", y="Age", color='Sex',
                  title='Athletes Agerage Age Trend')
    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(
                 type="buttons",
                 direction="up",
                 buttons=list([
                     dict(label="All Athletes",
                          method="update",
                          args=[{"visible": [True, True]},
                                {"title": "Averge Age of All Athletes Trend"}
                                ]),
                      dict(label="Female Average",
                           method="update",
                           args=[{"visible": [True, False]},
                                 {"title": 'Female Athletes Average Age \
                                 Trend', }
                                 ]),
                      dict(label="Male Average",
                           method="update",
                           args=[{"visible": [False, True]},
                                 {"title": "Male Athletes Average Age Trend", }
                                 ]),
                 ]),
            )
        ])

    fig.update_xaxes(dtick=4)
    fig.update_yaxes(dtick=1)
    # fig.show()
    fig.write_image('su_athletes_age_sex.png')


def continent_medals(olympics, continents):
    """
    Takes in the Olympic data and continent information data,
    use line graph to show the trend of medal wins by each
    continent, and use bar plot to show each year's average
    medal distribution in the same graph.
    """
    time = olympics['Year'] >= 1988
    season = olympics['Season'] == 'Summer'
    olympics = olympics[time & season & olympics.Medal.notnull()]
    # Drop out columns
    olympics = olympics[['Year', 'Team', 'Medal']]
    # Merge two dataframe
    merged = olympics.merge(continents, left_on='Team', right_on='name',
                            how='left')
    # Calculate medals in every year every continent
    result = merged.groupby(['Year',
                            'region'])['Medal'].count().reset_index()
    bar_res = result.groupby(['Year'], as_index=False)['Medal'].mean()
    # Plot
    fig = px.line(result, x="Year", y="Medal", color='region',
                  title='Continents Medals Attainment Trend')
    fig.add_bar(x=bar_res["Year"], y=bar_res["Medal"],
                name="Average Medals")
    # fig.show()
    fig.write_image('continent_medals.png')


def main():
    olympics = pd.read_csv('olympics.csv')
    continents = pd.read_csv('continents.csv')
    sport = pd.read_csv('sports_events.csv')
    top_10_teams_golds(olympics)
    top_5_summer_gold(olympics)
    events(olympics)
    most_medal_athlete(olympics)
    us_medals_by_sports(olympics, sport)
    su_athletes_age_sex(olympics)
    continent_medals(olympics, continents)


if __name__ == '__main__':
    main()
