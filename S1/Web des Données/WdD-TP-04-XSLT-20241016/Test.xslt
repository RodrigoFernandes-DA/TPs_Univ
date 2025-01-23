<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" indent="yes" />
    
    <xsl:template match="/">
        <html>
            <head>
                <title>Le Seigneur des Anneaux</title>
            </head>
            <body>
                <h1>Le Seigneur des Anneaux</h1>
                
                <!-- Lista de Filmes -->
                <h2>Films</h2>
                <ul>
                    <xsl:apply-templates select="lord-of-the-rings/films/film" />
                </ul>

                <!-- Lista de Atores -->
                <h2>Acteurs</h2>
                <ul>
                    <xsl:apply-templates select="lord-of-the-rings/actors/actor" />
                </ul>
            </body>
        </html>
    </xsl:template>

    <!-- Template para exibir cada filme -->
    <xsl:template match="film">
        <li>
            <h3><xsl:value-of select="title" /></h3>
            <p><b>Réalisateur:</b> <xsl:value-of select="director" /></p>
            <p><b>Date de sortie:</b> <xsl:value-of select="release-date" /></p>
            <p><b>Box-Office:</b> <xsl:value-of select="box-office" /> USD</p>
            <p><b>Budget:</b> <xsl:value-of select="budget" /> USD</p>
            <p><b>Rôles principaux:</b></p>
            <ul>
                <xsl:apply-templates select="roles/role" />
            </ul>
        </li>
    </xsl:template>

    <!-- Template para exibir cada papel principal (role) -->
    <xsl:template match="role">
        <li><xsl:value-of select="@character" /></li>
    </xsl:template>

    <!-- Template para exibir cada ator -->
    <xsl:template match="actor">
        <li>
            <h3><xsl:value-of select="name" /></h3>
            <p><b>Date de naissance:</b> <xsl:value-of select="brith-date" /></p>
            <p><b>Nationalité:</b> <xsl:value-of select="nationality" /></p>
            <p><a href="{wiki-page}" target="_blank">Page Wiki</a></p>
        </li>
    </xsl:template>
</xsl:stylesheet>
